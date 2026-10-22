# KoMaP 2026 - training (train 70 only, valid는 checkpoint 선택에만 사용)
# python train.py --data DATA --all          (10 models)
# python train.py --data DATA --name L768_s7
import os, json, time, random, argparse, gc, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize, dilation, diamond
from PIL import Image
from komap_lib import (load_pairs, upscale, build, load_model, predict_prob, predict_prob_s, remove_absent,
                       dataset_miou, seed_all, thickness, feature_scale, resize_s)

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.')
ap.add_argument('--config', default='configs.json')
ap.add_argument('--out', default='weights_retrained')
ap.add_argument('--name', default=None)
ap.add_argument('--all', action='store_true')
args = ap.parse_args()
cfgs = json.load(open(args.config, encoding='utf-8'))
names = cfgs['final_models'] if args.all else [args.name]
names = sorted(names, key=lambda n: cfgs['models'][n].get('scale_mode', 'size') == 'feature')   # feature 모델은 size 모델 필요
device, AMP = 'cuda', True

train_pairs, _ = load_pairs(args.data, 'train')
val_pairs, _ = load_pairs(args.data, 'valid')
val_imgs = [i for i, _ in val_pairs]
gts = [g for _, g in val_pairs]

def make_skel(g):
    sk = np.full(g.shape, 255, np.uint8)
    for c in (1, 2, 3):
        m = g == c
        if m.any():
            sk[dilation(skeletonize(m), diamond(2)).astype(bool) & m] = c
    return sk

_DATA = {}
def get_data(scale_mode):
    # size: x2 / feature: 이미지별 s = target / thickness(GT)
    if scale_mode in _DATA:
        return _DATA[scale_mode]
    if scale_mode == 'size':
        d = [(upscale(i), upscale(g, resample=Image.NEAREST)) for i, g in train_pairs]
    else:
        FS = cfgs['feature_scale']
        ss = [feature_scale(thickness(g == 1), FS['target'], FS.get('s_min', 1.5), FS.get('s_max', 4.0)) for _, g in train_pairs]
        print(f'  train scale: min {min(ss):.2f} / median {np.median(ss):.2f} / max {max(ss):.2f}')
        d = [(resize_s(i, s), resize_s(g, s, Image.NEAREST)) for (i, g), s in zip(train_pairs, ss)]
    _DATA[scale_mode] = [(bi, bg, make_skel(bg)) for bi, bg in d]
    return _DATA[scale_mode]

_VS = None
def feature_val_scales():
    # valid scale: size 모델 앙상블 예측으로 두께 측정 (GT 미사용)
    global _VS
    if _VS is not None:
        return _VS
    FS = cfgs['feature_scale']
    size_m = [n for n in cfgs['final_models'] if cfgs['models'][n].get('scale_mode', 'size') == 'size']
    acc = [0] * len(val_imgs)
    for n in size_m:
        p = os.path.join(args.out, f'{n}.pth')
        p = p if os.path.exists(p) else os.path.join('weights', f'{n}.pth')
        c = cfgs['models'][n]
        m = load_model(p, c.get('arch', 'unet'), c.get('encoder', 'resnet34'), c.get('attention', 'scse'))
        acc = [a + predict_prob(m, im, tta=8, tile=1024) for a, im in zip(acc, val_imgs)]
        del m; gc.collect(); torch.cuda.empty_cache()
    _VS = [feature_scale(thickness(remove_absent(a / len(size_m)) == 1), FS['target'], FS.get('s_min', 1.5), FS.get('s_max', 4.0)) for a in acc]
    print('  valid scale:', ', '.join(f'{s:.2f}' for s in _VS))
    return _VS

class CropDataset(Dataset):
    # random crop + rot/flip + contrast/brightness/gamma
    def __init__(self, data, crop=512, n_per_epoch=1000, gamma_log=False):
        self.data, self.crop, self.n, self.gamma_log = data, crop, n_per_epoch, gamma_log
    def __len__(self): return self.n
    def __getitem__(self, i):
        bi, bg, sk = random.choice(self.data)
        H, W = bg.shape; cr = self.crop
        if H < cr or W < cr:
            ph, pw = max(0, cr - H), max(0, cr - W)
            bi, bg, sk = (np.pad(a, ((0, ph), (0, pw)), mode='reflect') for a in (bi, bg, sk))
            H, W = bg.shape
        y = random.randint(0, H - cr); x = random.randint(0, W - cr)
        a = bi[y:y+cr, x:x+cr].astype(np.float32) / 255
        g = bg[y:y+cr, x:x+cr]; s = sk[y:y+cr, x:x+cr]
        k = random.randint(0, 3); a, g, s = np.rot90(a, k), np.rot90(g, k), np.rot90(s, k)
        if random.random() < 0.5:
            a, g, s = np.fliplr(a), np.fliplr(g), np.fliplr(s)
        c = random.uniform(0.8, 1.2); b = random.uniform(-0.1, 0.1)
        gm = float(np.exp(random.uniform(np.log(0.8), np.log(1.25)))) if self.gamma_log else random.uniform(0.8, 1.25)
        a = np.clip((a - 0.5) * c + 0.5 + b, 0, 1) ** gm
        return (torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).repeat(3, 1, 1),
                torch.from_numpy(np.ascontiguousarray(g)).long(),
                torch.from_numpy(np.ascontiguousarray(s)).long())

# loss: CE + Lovasz (+ optional skeleton recall)
ce = nn.CrossEntropyLoss(); lovasz = smp.losses.LovaszLoss(mode='multiclass')
def skel_recall(out, sk):
    p = out.float().softmax(1); ls = []
    for c in (1, 2, 3):
        m = (sk == c).float(); n = m.sum()
        if n > 0: ls.append(1 - (p[:, c] * m).sum() / n)
    return torch.stack(ls).mean() if ls else out.sum() * 0

def save_curve(rows, path, title):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception:
        return
    ep = [r[0] for r in rows]
    fig, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(ep, [r[1] for r in rows], color='tab:gray', label='train loss'); a1.set_xlabel('epoch'); a1.set_ylabel('train loss')
    a2 = a1.twinx(); a2.plot(ep, [r[2] for r in rows], color='tab:blue', marker='o', ms=3, label='valid mIoU'); a2.set_ylabel('valid mIoU (%)')
    b = max(rows, key=lambda r: r[2]); a2.scatter([b[0]], [b[2]], color='tab:red', zorder=5)
    a2.annotate(f'best {b[2]:.2f} @ ep {b[0]} (saved)', (b[0], b[2]), textcoords='offset points', xytext=(-60, -18), color='tab:red')
    plt.title(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
for n in names:
    c = cfgs['models'][n]
    assert c.get('loss', 'ce_lovasz') == 'ce_lovasz', f'unsupported loss: {c.get("loss")}'
    smode = c.get('scale_mode', 'size')
    epochs, bs, seed = c.get('epochs', 100), c.get('bs', 8), c.get('seed', 42)
    crop, val_tile, w_sk = c.get('crop', 512), c.get('val_tile', 512), c.get('skel', 0.0)
    data = get_data(smode)
    vs = feature_val_scales() if smode == 'feature' else None
    seed_all(seed)
    m = build(c.get('arch', 'unet'), c.get('encoder', 'resnet34'), c.get('attention', 'scse'), weights='imagenet').to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=c.get('lr', 3e-4))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    loader = DataLoader(CropDataset(data, crop, gamma_log=(smode == 'feature')), batch_size=bs, shuffle=True, num_workers=0)
    path = os.path.join(args.out, f'{n}.pth'); best = -1; t0 = time.time()
    log_path = os.path.join(args.out, 'logs', f'{n}_train_log.csv'); rows = []
    with open(log_path, 'w', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow(['epoch', 'train_loss', 'valid_miou', 'saved', 'minutes'])
    print(f'start {n} | {c}')
    for ep in range(1, epochs + 1):
        m.train(); tot = 0
        for x, y, s in loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            with torch.autocast(device_type='cuda', enabled=AMP):
                out = m(x); loss = ce(out, y) + lovasz(out, y)
            if w_sk > 0: loss = loss + w_sk * skel_recall(out, s)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sch.step()
        if ep % 5 == 0 or ep == epochs:   # 5 epoch마다 valid, best 저장
            if smode == 'feature':
                preds = [predict_prob_s(m, im, s, tta=1, tile=val_tile).argmax(0) for im, s in zip(val_imgs, vs)]
            else:
                preds = [predict_prob(m, im, tta=1, tile=val_tile).argmax(0) for im in val_imgs]
            sc = dataset_miou(preds, gts)
            mark = ''
            if sc > best: best = sc; torch.save(m.state_dict(), path); mark = ' *saved'
            print(f'  {n} [{ep}/{epochs}] loss {tot/len(loader):.3f} | val {sc:.2f}{mark} ({(time.time()-t0)/60:.0f} min)')
            rows.append((ep, tot / len(loader), sc))
            with open(log_path, 'a', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerow([ep, f'{tot/len(loader):.4f}', f'{sc:.4f}', int(bool(mark)), f'{(time.time()-t0)/60:.1f}'])
    save_curve(rows, os.path.join(args.out, 'logs', f'{n}_train_curve.png'), f'{n} ({smode})')
    print(f'done {n}: best {best:.2f} -> {path}')
    del m, opt; gc.collect(); torch.cuda.empty_cache()
