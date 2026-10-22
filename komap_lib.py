# KoMaP 2026 - common utils
import os, glob, random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

COLORS = {0: (153, 127, 76), 1: (204, 204, 204), 2: (76, 178, 76), 3: (25, 76, 153)}
NAMES = ['Al', 'Eutectic Si', 'Primary Si', 'Al3Ni']


def rgb_to_index(m):
    out = np.full(m.shape[:2], 255, dtype=np.uint8)
    for k, c in COLORS.items():
        out[np.all(m == c, axis=-1)] = k
    return out


def index_to_rgb(idx):
    out = np.zeros((*idx.shape, 3), dtype=np.uint8)
    for k, c in COLORS.items():
        out[idx == k] = c
    return out


# mIoU: 이미지별 평균 후 전체 평균 (GT/pred 모두 없는 클래스는 제외)
def image_miou(pred, gt, n_classes=4):
    ious = []
    for k in range(n_classes):
        p, g = (pred == k), (gt == k)
        union = (p | g).sum()
        ious.append(np.nan if union == 0 else (p & g).sum() / union)
    return np.nanmean(ious), ious


def dataset_miou(preds, gts):
    return float(np.mean([image_miou(p, g)[0] for p, g in zip(preds, gts)]) * 100)


def per_class_iou(preds, gts):
    return np.nanmean(np.array([image_miou(p, g)[1] for p, g in zip(preds, gts)], dtype=float), axis=0) * 100


def load_pairs(root, split):
    pairs, names = [], []
    for mp in sorted(glob.glob(os.path.join(root, split, 'masks', '*.png'))):
        ip = mp.replace('masks', 'images').replace('_mask', '_image')
        pairs.append((np.array(Image.open(ip).convert('L')),
                      rgb_to_index(np.array(Image.open(mp).convert('RGB')))))
        names.append(os.path.basename(mp))
    return pairs, names


def load_images(root, split):
    files = sorted(glob.glob(os.path.join(root, split, 'images', '*.png')))
    return [np.array(Image.open(f).convert('L')) for f in files], files


def upscale(img, scale=2, resample=Image.BICUBIC):
    H, W = img.shape
    return np.array(Image.fromarray(img).resize((W * scale, H * scale), resample))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build(arch='unet', encoder='resnet34', attention='scse', weights=None):
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=4)
    if arch == 'unet':
        return smp.Unet(decoder_attention_type=attention, **kw)
    if arch == 'unetpp':
        return smp.UnetPlusPlus(decoder_attention_type=attention, **kw)
    raise ValueError(arch)


def load_model(path, arch='unet', encoder='resnet34', attention='scse', device='cuda'):
    m = build(arch, encoder, attention).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def tta_ops(tta):
    if tta == 8: return [(k, f) for k in range(4) for f in (False, True)]
    if tta == 4: return [(0, False), (0, True), (2, False), (2, True)]
    return [(0, False)]


# sliding window (overlap = tile/8) + TTA, 결과는 원본 크기로 area 다운샘플
@torch.no_grad()
def tiled_prob(m, big, H, W, tta=8, tile=1024, device='cuda', amp=True):
    overlap = tile // 8
    H2, W2 = big.shape
    x_all = torch.from_numpy(big).float().div(255)[None, None].repeat(1, 3, 1, 1).to(device)
    prob = torch.zeros(1, 4, H2, W2, device=device)
    cnt = torch.zeros(1, 1, H2, W2, device=device)
    step = tile - overlap
    ys = list(range(0, max(H2 - tile, 0) + 1, step)); xs = list(range(0, max(W2 - tile, 0) + 1, step))
    if ys[-1] + tile < H2: ys.append(H2 - tile)
    if xs[-1] + tile < W2: xs.append(W2 - tile)
    ops = tta_ops(tta)
    for y0 in ys:
        for x0 in xs:
            xt = x_all[:, :, y0:y0 + tile, x0:x0 + tile]
            th, tw = xt.shape[2:]
            ph, pw = (32 - th % 32) % 32, (32 - tw % 32) % 32
            xt = F.pad(xt, (0, pw, 0, ph), mode='reflect')
            p = 0
            for k, fl in ops:
                xf = torch.rot90(xt, k, dims=(2, 3))
                if fl: xf = torch.flip(xf, [3])
                with torch.autocast(device_type='cuda', enabled=amp):
                    out = m(xf).float().softmax(1)
                if fl: out = torch.flip(out, [3])
                p = p + torch.rot90(out, -k, dims=(2, 3))
            prob[:, :, y0:y0 + th, x0:x0 + tw] += (p / len(ops))[:, :, :th, :tw]
            cnt[:, :, y0:y0 + th, x0:x0 + tw] += 1
    prob = F.interpolate(prob / cnt, size=(H, W), mode='area')
    return prob[0].cpu().numpy()


def predict_prob(m, img, tta=8, tile=1024, scale=2, device='cuda'):
    m.eval()
    H, W = img.shape
    return tiled_prob(m, upscale(img, scale), H, W, tta=tta, tile=tile, device=device)


# primary Si가 0.5% 미만이면 오검출로 보고 제거 (train에서 존재 시 최소 1.53%)
def remove_absent(prob, min_frac=0.005, classes=(2,)):
    pred = prob.argmax(0)
    for c in classes:
        if 0 < (pred == c).mean() < min_frac:
            q = prob.copy(); q[c] = -1
            pred = q.argmax(0)
    return pred


# eutectic Si 평균 두께(px): skeleton 위 EDT x 2
def thickness(mask):
    from scipy import ndimage
    from skimage.morphology import skeletonize
    if mask.sum() < 200:
        return float('nan')
    dt = ndimage.distance_transform_edt(mask)
    v = dt[skeletonize(mask)]
    return float(np.mean(2 * v)) if v.size else float('nan')


def feature_scale(t, target, s_min=1.5, s_max=4.0):
    return 2.0 if not np.isfinite(t) else float(np.clip(target / t, s_min, s_max))


def resize_s(a, s, resample=Image.BICUBIC):
    H, W = a.shape
    return np.array(Image.fromarray(a).resize((max(1, round(W * s)), max(1, round(H * s))), resample))


def predict_prob_s(m, img, s, tta=8, tile=1024, device='cuda'):
    m.eval()
    H, W = img.shape
    return tiled_prob(m, resize_s(img, s), H, W, tta=tta, tile=tile, device=device)
