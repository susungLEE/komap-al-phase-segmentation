# KoMaP 2026 - inference / reproduction
# python predict.py --data DATA --weights weights --out submission_reproduced [--compare SUBMITTED_MASK_DIR]
import os, json, time, argparse, gc, platform, datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from komap_lib import (NAMES, COLORS, load_pairs, load_images, load_model, predict_prob, predict_prob_s, remove_absent,
                       dataset_miou, per_class_iou, index_to_rgb, rgb_to_index, thickness, feature_scale)

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.')
ap.add_argument('--weights', default='weights')
ap.add_argument('--config', default='configs.json')
ap.add_argument('--out', default='submission_reproduced')
ap.add_argument('--compare', default=None)
ap.add_argument('--tile', type=int, default=1024)
ap.add_argument('--tta', type=int, default=8)
ap.add_argument('--small_w', type=int, default=800)
ap.add_argument('--no_vis', action='store_true')
ap.add_argument('--small_scales', default='2,3,4')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, 'metrics.txt'), 'w', encoding='utf-8')
def log(s=''):
    print(s, flush=True)
    LOG.write(s + '\n'); LOG.flush()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
log(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {GPU} | Python {platform.python_version()} | PyTorch {torch.__version__}')
torch.backends.cudnn.benchmark = False
cfgs = json.load(open(args.config, encoding='utf-8'))
MODELS = cfgs['final_models']
FS = cfgs.get('feature_scale', {})
SMALL = [int(s) for s in args.small_scales.split(',')]
mode = lambda n: cfgs['models'][n].get('scale_mode', 'size')
SIZE_M = [n for n in MODELS if mode(n) == 'size']      # x2 학습 모델
FEAT_M = [n for n in MODELS if mode(n) == 'feature']   # 조직 두께 기준 배율 학습 모델

val_pairs, val_files = load_pairs(args.data, 'valid')
val_imgs, gts = [p[0] for p in val_pairs], [p[1] for p in val_pairs]
test_imgs, test_files = load_images(args.data, 'test')
log(f'valid {len(val_imgs)}, test {len(test_imgs)} | models {len(MODELS)} (size {len(SIZE_M)} + feature {len(FEAT_M)}) | tile {args.tile}, TTA {args.tta}')

M = {'run_time': f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}', 'device': GPU,
     'settings': {'tile': args.tile, 'tta': args.tta, 'small_w': args.small_w, 'small_scales': SMALL,
                  'n_models': len(MODELS), 'size_models': SIZE_M, 'feature_models': FEAT_M},
     'single_model_val_miou': {}}

def load(n):
    c = cfgs['models'][n]
    return load_model(os.path.join(args.weights, f'{n}.pth'), c.get('arch', 'unet'), c.get('encoder', 'resnet34'),
                      c.get('attention', 'scse'), device=DEVICE)

IMGS = {'v': val_imgs, 't': test_imgs}
acc = {k: [0] * len(v) for k, v in IMGS.items()}    # ensemble prob sum
ref2 = {k: [0] * len(v) for k, v in IMGS.items()}   # size 모델 x2 예측 합 (두께 측정용)
t0 = time.time()

# size models: x2, 작은 이미지(width < 800)는 x2/x3/x4 평균
for n in SIZE_M:
    m = load(n)
    per = {}
    for k, imgs in IMGS.items():
        ps = []
        for i, im in enumerate(imgs):
            p2 = predict_prob(m, im, args.tta, args.tile, scale=2, device=DEVICE)
            ref2[k][i] = ref2[k][i] + p2
            if im.shape[1] < args.small_w:
                p = sum(p2 if s == 2 else predict_prob(m, im, args.tta, args.tile, scale=s, device=DEVICE) for s in SMALL) / len(SMALL)
            else:
                p = p2
            ps.append(p)
        acc[k] = [a + p for a, p in zip(acc[k], ps)]
        per[k] = ps
    s = dataset_miou([remove_absent(p) for p in per['v']], gts)
    M['single_model_val_miou'][n] = round(float(s), 4)
    log(f'  {n:<22s} [size]    val mIoU {s:.2f}  ({(time.time()-t0)/60:.1f} min)')
    del m, per; gc.collect(); torch.cuda.empty_cache()

# feature models: 1차 예측으로 eutectic Si 두께 측정 -> 목표 두께가 되도록 배율 결정 (GT 미사용)
if FEAT_M:
    T = FS['target']; smin, smax = FS.get('s_min', 1.5), FS.get('s_max', 4.0)
    SC = {k: [feature_scale(thickness(remove_absent(r / len(SIZE_M)) == 1), T, smin, smax) for r in ref2[k]] for k in IMGS}
    log(f'  scale (target {T:.2f}px) valid: ' + ', '.join(f'{s:.2f}' for s in SC['v']))
    log(f'  scale (target {T:.2f}px) test : ' + ', '.join(f'{s:.2f}' for s in SC['t']))
    M['feature_scale'] = {'target_thickness_px': T, 's_min': smin, 's_max': smax,
                          'valid': {os.path.basename(f): round(float(s), 3) for f, s in zip(val_files, SC['v'])},
                          'test': {os.path.basename(f): round(float(s), 3) for f, s in zip(test_files, SC['t'])}}

for n in FEAT_M:
    m = load(n)
    per = {}
    for k, imgs in IMGS.items():
        ps = [predict_prob_s(m, im, s, args.tta, args.tile, device=DEVICE) for im, s in zip(imgs, SC[k])]
        acc[k] = [a + p for a, p in zip(acc[k], ps)]
        per[k] = ps
    s = dataset_miou([remove_absent(p) for p in per['v']], gts)
    M['single_model_val_miou'][n] = round(float(s), 4)
    log(f'  {n:<22s} [feature] val mIoU {s:.2f}  ({(time.time()-t0)/60:.1f} min)')
    del m, per; gc.collect(); torch.cuda.empty_cache()

# ensemble
K = len(MODELS)
v_pred = [remove_absent(a / K) for a in acc['v']]
ens = dataset_miou(v_pred, gts)
per = per_class_iou(v_pred, gts)
log(f'\nEnsemble val mIoU {ens:.2f} | ' + ' / '.join(f'{NAMES[i]} {per[i]:.1f}' for i in range(4)))
M['ensemble_val_miou'] = round(float(ens), 4)
M['ensemble_val_class_iou'] = {NAMES[i]: round(float(per[i]), 4) for i in range(4)}

M['ensemble_val_miou_per_image'] = {}
log('  per image:')
for f, p, g in zip(val_files, v_pred, gts):
    s = dataset_miou([p], [g])
    M['ensemble_val_miou_per_image'][os.path.basename(f)] = round(float(s), 4)
    log(f'    {os.path.basename(f):<30s} {s:.2f}')

# confusion matrix (row = GT, col = pred), pixel-level precision / recall / F1
cm = np.zeros((4, 4), np.int64)
for p, g in zip(v_pred, gts):
    cm += np.bincount(g.ravel().astype(np.int64) * 4 + p.ravel(), minlength=16).reshape(4, 4)
cm_pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
M['confusion_matrix_pct'] = {NAMES[i]: {NAMES[j]: round(float(cm_pct[i, j]), 3) for j in range(4)} for i in range(4)}
log('\n  confusion matrix (%, row = GT, col = pred):')
log('    ' + ' ' * 13 + ''.join(f'{n:>13s}' for n in NAMES))
for i in range(4):
    log(f'    {NAMES[i]:<13s}' + ''.join(f'{cm_pct[i, j]:13.2f}' for j in range(4)))

tp = np.diag(cm).astype(float)
prec = tp / np.maximum(cm.sum(0), 1) * 100
rec = tp / np.maximum(cm.sum(1), 1) * 100
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
M['precision_recall_f1_pct'] = {NAMES[i]: {'precision': round(float(prec[i]), 3), 'recall': round(float(rec[i]), 3),
                                           'f1': round(float(f1[i]), 3)} for i in range(4)}
M['pixel_accuracy_pct'] = round(float(tp.sum() / cm.sum() * 100), 3)
log('\n  precision / recall / F1 (%):')
for i in range(4):
    log(f'    {NAMES[i]:<13s} P {prec[i]:6.2f} | R {rec[i]:6.2f} | F1 {f1[i]:6.2f}')
log(f'    pixel accuracy {M["pixel_accuracy_pct"]:.2f}%')


# ---- visualization ----
def _font(sz):
    for f in ['arial.ttf', 'DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf']:
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

def _panel(arr, title, w):
    im = Image.fromarray(arr).convert('RGB')
    h = int(im.height * w / im.width)
    im = im.resize((w, h), Image.NEAREST if arr.ndim == 3 else Image.BILINEAR)
    c = Image.new('RGB', (w, h + 34), 'white')
    c.paste(im, (0, 34))
    ImageDraw.Draw(c).text((6, 6), title, fill='black', font=_font(20))
    return c

def _overlay(im, idx, a=0.6):   # Al은 칠하지 않음
    g = np.stack([im] * 3, -1).astype(np.float32)
    c = index_to_rgb(idx).astype(np.float32); m = (idx > 0)[..., None]
    return np.where(m, g * (1 - a) + c * a, g).astype(np.uint8)

def _crop_box(shape, c=448):     # center crop
    H, W = shape; c = min(c, H, W)
    return (H - c) // 2, (W - c) // 2, c

def _grid(items, path, title, cols, cell=360):
    rows_ = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * (cell + 6), rows_ * (cell + 30) + 90), 'white'); d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, fill='black', font=_font(24))
    for k, (arr, cap) in enumerate(items):
        x, y = (k % cols) * (cell + 6), 44 + (k // cols) * (cell + 30)
        sheet.paste(Image.fromarray(arr).resize((cell, cell), Image.NEAREST), (x, y + 26))
        d.text((x + 4, y + 4), cap, fill='black', font=_font(16))
    x = 8; yb = sheet.height - 36
    for k, n in enumerate(NAMES):
        d.rectangle([x, yb, x + 22, yb + 22], fill=COLORS[k]); d.text((x + 28, yb), n, fill='black', font=_font(18)); x += 190
    sheet.save(path)

if not args.no_vis:
    os.makedirs(os.path.join(args.out, 'valid_pred'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'valid_compare'), exist_ok=True)
    for f, im, p, g in zip(val_files, val_imgs, v_pred, gts):
        name = os.path.basename(f)
        Image.fromarray(index_to_rgb(p)).save(os.path.join(args.out, 'valid_pred', name))
        err = np.stack([im // 2] * 3, -1).astype(np.uint8)
        err[p != g] = (255, 0, 0)
        s1 = dataset_miou([p], [g])
        W = 700
        ps = [_panel(im, 'Image', W), _panel(index_to_rgb(g), 'Ground truth', W),
              _panel(index_to_rgb(p), f'Prediction (mIoU {s1:.1f})', W),
              _panel(err, f'Errors in red ({(p != g).mean() * 100:.1f}% px)', W)]
        H = ps[0].height
        sheet = Image.new('RGB', (W * 2 + 10, H * 2 + 50), 'white')
        for k, q in enumerate(ps):
            sheet.paste(q, ((k % 2) * (W + 10), (k // 2) * (H + 10)))
        d = ImageDraw.Draw(sheet); x = 6
        for k, n in enumerate(NAMES):
            d.rectangle([x, H * 2 + 22, x + 22, H * 2 + 44], fill=COLORS[k]); d.text((x + 28, H * 2 + 22), n, fill='black', font=_font(20)); x += 200
        sheet.save(os.path.join(args.out, 'valid_compare', name.replace('_mask', '_compare')))

    cs = 150; pad = 160
    img = Image.new('RGB', (pad + cs * 4 + 40, pad + cs * 4 + 60), 'white'); d = ImageDraw.Draw(img)
    for i in range(4):
        for j in range(4):
            v = cm_pct[i, j]; t = int(255 - v * 2.2)
            d.rectangle([pad + j * cs, pad + i * cs, pad + (j + 1) * cs - 2, pad + (i + 1) * cs - 2], fill=(t, t, 255))
            d.text((pad + j * cs + 40, pad + i * cs + 60), f'{v:.1f}%', fill='white' if v > 50 else 'black', font=_font(24))
        d.text((10, pad + i * cs + 60), NAMES[i], fill='black', font=_font(20))
        d.text((pad + i * cs + 10, pad - 40), NAMES[i], fill='black', font=_font(20))
    d.text((10, 20), 'Confusion matrix (row: truth, column: prediction)', fill='black', font=_font(22))
    d.text((pad, pad + cs * 4 + 15), f'Validation 20 images, ensemble mIoU {ens:.2f}', fill='black', font=_font(20))
    img.save(os.path.join(args.out, 'confusion_matrix.png'))

    tr_pairs, tr_files = load_pairs(args.data, 'train')
    pick = np.random.default_rng(0).choice(len(tr_pairs), min(16, len(tr_pairs)), replace=False)
    items = []
    for i in sorted(pick):
        im, g = tr_pairs[i]; y0, x0, c = _crop_box(g.shape)
        items.append((_overlay(im[y0:y0+c, x0:x0+c], g[y0:y0+c, x0:x0+c]), os.path.basename(tr_files[i]).replace('_mask.png', '')))
    _grid(items, os.path.join(args.out, 'train_labels_grid.png'), 'Train labels (16 of 70, center crop)', 4)
    del tr_pairs

    lab, prd = [], []
    for f, im, p, g in zip(val_files, val_imgs, v_pred, gts):
        y0, x0, c = _crop_box(g.shape); nm = os.path.basename(f).replace('_mask.png', '')
        lab.append((_overlay(im[y0:y0+c, x0:x0+c], g[y0:y0+c, x0:x0+c]), nm))
        prd.append((_overlay(im[y0:y0+c, x0:x0+c], p[y0:y0+c, x0:x0+c]), f'{nm}  mIoU {dataset_miou([p], [g]):.1f}'))
    _grid(lab, os.path.join(args.out, 'val_labels_grid.png'), 'Validation labels (center crop)', 5)
    _grid(prd, os.path.join(args.out, 'val_pred_grid.png'), f'Validation predictions (mIoU {ens:.2f}, same crops)', 5)
    log('  saved: valid_pred/, valid_compare/, confusion_matrix.png, *_grid.png')

# test masks
for f, a in zip(test_files, acc['t']):
    Image.fromarray(index_to_rgb(remove_absent(a / K))).save(
        os.path.join(args.out, os.path.basename(f).replace('_image', '_mask')))
log(f'\ntest masks {len(test_files)} -> {args.out}  ({(time.time()-t0)/60:.1f} min)')
M['test_masks_saved'] = len(test_files)
M['elapsed_min'] = round((time.time() - t0) / 60, 2)

if args.compare:
    agree = {}
    for f in test_files:
        name = os.path.basename(f).replace('_image', '_mask')
        a = rgb_to_index(np.array(Image.open(os.path.join(args.out, name)).convert('RGB')))
        b = rgb_to_index(np.array(Image.open(os.path.join(args.compare, name)).convert('RGB')))
        agree[name] = round(float((a == b).mean() * 100), 4)
    v = list(agree.values())
    log(f'agreement with {args.compare}: mean {np.mean(v):.3f}% (min {np.min(v):.3f}%)')
    M['compare'] = {'target': args.compare, 'mean_agreement_pct': round(float(np.mean(v)), 4),
                    'min_agreement_pct': round(float(np.min(v)), 4), 'per_image': agree}

json.dump(M, open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
log('saved: metrics.json, metrics.txt')
LOG.close()
