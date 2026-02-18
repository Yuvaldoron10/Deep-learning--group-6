
"""
IFCNN Stage-2 TRAIN+VAL (together)

- Imports model from: model_finetune.py
- Loads baseline weights: snapshots/IFCNN-MAX.pth
- Trains up to 30 epochs (SGD + poly LR)
- Loss = MSE + lambda * PerceptualLoss (ResNet101 loaded from local)
- After EACH epoch:
    1) compute TRAIN avg loss
    2) compute VAL avg loss
    3) update loss plot (two curves: train & val)
    4) run VAL inference

- Saves weights EVERY epoch
"""

import os
import glob
import csv
import time
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
plt.ion()
from model_finetune import myIFCNN


try:
    import piq
    _HAS_PIQ = True
except Exception:
    _HAS_PIQ = False


# =========================================================
# 0) Device
# =========================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print("Using device:", device)
print("torch:", torch.__version__)
print("piq available:", _HAS_PIQ)


# =========================================================
# 1) Paths
# =========================================================
EXPERIMENT_NAME = "model_original"
MODEL_TAG = "IFCNN-MAX-original"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(SCRIPT_DIR, "data-image-splits")
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
VAL_ROOT   = os.path.join(DATA_ROOT, "val")

TRAIN_MULTI_ROOT = os.path.join(TRAIN_ROOT, "multi focus")
TRAIN_GT_ROOT    = os.path.join(TRAIN_ROOT, "Ground_truth")

VAL_MULTI_ROOT = os.path.join(VAL_ROOT, "multi focus")
VAL_GT_ROOT    = os.path.join(VAL_ROOT, "Ground_truth")

SNAP_DIR = os.path.join(SCRIPT_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

BASELINE_CKPT = os.path.join(SNAP_DIR, "IFCNN-MAX.pth")
RESNET101_CKPT = os.path.join(SNAP_DIR, "resnet101-63fe2227.pth")

assert os.path.isfile(BASELINE_CKPT), f"Missing baseline: {BASELINE_CKPT}"
assert os.path.isfile(RESNET101_CKPT), f"Missing local ResNet101 weights: {RESNET101_CKPT}"


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_val", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

SAVE_DIR = RESULTS_DIR
CSV_METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_vs_gt.csv")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "loss_curve.png")
LOSS_HISTORY_CSV = os.path.join(RESULTS_DIR, "loss_history.csv")
METRICS_HISTORY_CSV = os.path.join(RESULTS_DIR, "metrics_history.csv")
METRICS_PLOT_PATH   = os.path.join(RESULTS_DIR, "metrics_curve.png")


CKPT_PREFIX = os.path.join(SNAP_DIR, f"{MODEL_TAG}")
CKPT_LAST   = os.path.join(SNAP_DIR, f"{MODEL_TAG}_last.pth")

# =========================================================
# 2) Hyperparameters
# =========================================================
EPOCHS = 30
BATCH_SIZE = 1
ACCUM_STEPS = 32
LR = 0.01

# LR = 1e-4
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

LAMBDA_MSE = 1.0
LAMBDA_PER = 1.0


SAVE_EVERY = 1
NUM_WORKERS = 0


# =========================================================
# 3) Transforms (ImageNet normalize)
# =========================================================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

mean_t = torch.tensor(MEAN).view(3, 1, 1)
std_t  = torch.tensor(STD).view(3, 1, 1)


def denorm_to_uint8(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().cpu()
    x = x * std_t + mean_t
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


# =========================================================
# 4) Dataset utilities
# =========================================================
def center_crop_pil(img: Image.Image, h: int, w: int) -> Image.Image:
    W, H = img.size
    left = max((W - w) // 2, 0)
    top  = max((H - h) // 2, 0)
    return img.crop((left, top, left + w, top + h))


def pick_gt_image_for_scene(scene_gt_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(scene_gt_dir, "**", "*.png"),
        os.path.join(scene_gt_dir, "**", "*.jpg"),
        os.path.join(scene_gt_dir, "**", "*.jpeg"),
        os.path.join(scene_gt_dir, "**", "*.bmp"),
        os.path.join(scene_gt_dir, "**", "*.tif"),
        os.path.join(scene_gt_dir, "**", "*.tiff"),
        os.path.join(scene_gt_dir, "**", "*.webp"),
    ]
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p, recursive=True))
    cands = sorted([c for c in cands if os.path.isfile(c)])
    return cands[0] if len(cands) else None


class SceneDataset(Dataset):
    def __init__(self, multi_root: str, gt_root: str):
        self.multi_root = multi_root
        self.gt_root = gt_root

        self.scenes = []
        for s in sorted(os.listdir(multi_root)):
            in_dir = os.path.join(multi_root, s)
            gt_dir = os.path.join(gt_root, s)
            if not os.path.isdir(in_dir) or not os.path.isdir(gt_dir):
                continue

            in_imgs = sorted(glob.glob(os.path.join(in_dir, "*.*")))
            gt_img = pick_gt_image_for_scene(gt_dir)

            if len(in_imgs) >= 2 and gt_img is not None:
                self.scenes.append(s)

        print(f"[DATA] Found {len(self.scenes)} scenes in: {multi_root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]

        in_dir = os.path.join(self.multi_root, scene)
        gt_dir = os.path.join(self.gt_root, scene)

        in_paths = sorted([p for p in glob.glob(os.path.join(in_dir, "*.*")) if os.path.isfile(p)])
        if len(in_paths) == 0:
            raise RuntimeError(f"No input images in: {in_dir}")

        gt_path = pick_gt_image_for_scene(gt_dir)
        if gt_path is None:
            raise RuntimeError(f"No GT found under: {gt_dir}")

        in_imgs = [Image.open(p).convert("RGB") for p in in_paths]
        gt_img  = Image.open(gt_path).convert("RGB")

        hs = [im.size[1] for im in in_imgs] + [gt_img.size[1]]
        ws = [im.size[0] for im in in_imgs] + [gt_img.size[0]]
        h, w = min(hs), min(ws)

        in_imgs = [center_crop_pil(im, h, w) for im in in_imgs]
        gt_img  = center_crop_pil(gt_img, h, w)

        in_tensors = [tf(im) for im in in_imgs]
        gt_tensor  = tf(gt_img)

        return in_tensors, gt_tensor, scene


def collate_scene(batch):
    xs, ys, scenes = zip(*batch)  # xs: tuple of lists, ys: tuple of tensors

    B = len(xs)
    n_inputs = len(xs[0])

    gt_batched = torch.stack(ys, dim=0)


    inputs_batched = []
    for k in range(n_inputs):
        inputs_batched.append(torch.stack([xs[b][k] for b in range(B)], dim=0))

    return inputs_batched, gt_batched, list(scenes)


# =========================================================
# 5) Perceptual Net (LOCAL ResNet101)
# =========================================================
class ResNet101Perceptual(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()
        net = models.resnet101(weights=None)
        sd = torch.load(weights_path, map_location="cpu")

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        net.load_state_dict(sd, strict=True)



        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


mse_loss = nn.MSELoss()


def perceptual_loss(perc_net: nn.Module, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        f_gt = perc_net(gt)
    f_pred = perc_net(pred)
    return mse_loss(f_pred, f_gt)


# =========================================================
# 6) LR schedule (poly)
# =========================================================
def poly_lr(base_lr: float, cur: int, max_iter: int, power: float) -> float:
    return base_lr * (1 - cur / max_iter) ** power


# =========================================================
# 7) Metrics helpers
# =========================================================
def align_same_size_np(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)

    def crop_center(x, h, w):
        H, W = x.shape[:2]
        top = max((H - h) // 2, 0)
        left = max((W - w) // 2, 0)
        return x[top:top + h, left:left + w]

    return crop_center(a, h, w), crop_center(b, h, w)


def to_gray_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[..., 0].astype(np.float32)
    g = rgb_u8[..., 1].astype(np.float32)
    b = rgb_u8[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).round().astype(np.uint8)


def psnr_uint8(img_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    img = img_u8.astype(np.float32)
    ref = ref_u8.astype(np.float32)
    mse = np.mean((img - ref) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def ssim_gray(gray_u8: np.ndarray, ref_gray_u8: np.ndarray) -> float:
    return float(ssim(gray_u8, ref_gray_u8, data_range=255))


def vif_p_optional(fused_gray_u8: np.ndarray, gt_gray_u8: np.ndarray) -> float:
    if not _HAS_PIQ:
        return float("nan")
    f = torch.from_numpy(fused_gray_u8).float() / 255.0
    g = torch.from_numpy(gt_gray_u8).float() / 255.0
    f = f.unsqueeze(0).unsqueeze(0)
    g = g.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        v = piq.vif_p(f, g, data_range=1.0)
    return float(v.item())


def mean_ignore_nan(vals: List[float]) -> float:
    good = [v for v in vals if isinstance(v, (float, int)) and not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.mean(good)) if len(good) else float("nan")


# =========================================================
# 8) Plot Loss (SAVE + SHOW each epoch)
# =========================================================
def plot_loss_curve(epochs, train_losses, val_losses, save_path):
    plt.clf()
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.pause(0.001)

def minmax01(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-12:
        return np.zeros_like(x)
    return (x - m) / (M - m)


def plot_metrics_curve(epochs, viff_means, issim_means, psnr_means, save_path):
    v = minmax01(viff_means)
    s = minmax01(issim_means)
    p = minmax01(psnr_means)

    plt.clf()
    plt.plot(epochs, v, label="VIFF (norm)", marker="o")
    plt.plot(epochs, s, label="SSIM (norm)", marker="o")
    plt.plot(epochs, p, label="PSNR (norm)", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Normalized value [0–1]")
    plt.title("Validation Metrics (Normalized) vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.pause(0.001)

# =========================================================
# 9) One-epoch train / val loss
# =========================================================
def run_one_epoch_train(model, perc_net, loader, optimizer, cur_update, total_updates):
    model.train()

    def set_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    model.train()
    model.apply(set_bn_eval)
    running = 0.0

    optimizer.zero_grad(set_to_none=True)

    micro_step = 0
    for inputs, gt, _scene in loader:
        micro_step += 1

        inputs = [x.to(device) for x in inputs]
        gt = gt.to(device)

        pred = model(*inputs)

        l_mse = mse_loss(pred, gt)
        l_per = perceptual_loss(perc_net, pred, gt)
        loss = LAMBDA_MSE * l_mse + LAMBDA_PER * l_per

        (loss / ACCUM_STEPS).backward()

        running += float(loss.item())

        do_step = (micro_step % ACCUM_STEPS == 0)
        last_batch = (micro_step == len(loader))

        if do_step or last_batch:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            cur_update += 1
            lr_now = poly_lr(LR, cur_update, total_updates, POLY_POWER)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

    avg = running / max(1, len(loader))
    return avg, cur_update


@torch.no_grad()
def run_one_epoch_val_loss(model, perc_net, loader):
    model.eval()
    running = 0.0

    for inputs, gt, _scenes in loader:
        inputs = [x.to(device) for x in inputs]
        gt = gt.to(device)

        pred = model(*inputs)

        l_mse = mse_loss(pred, gt)
        l_per = perceptual_loss(perc_net, pred, gt)
        loss = LAMBDA_MSE * l_mse + LAMBDA_PER * l_per

        running += float(loss.item())

    avg = running / max(1, len(loader))
    return avg


# =========================================================
# 10) Validation inference
# =========================================================
@torch.no_grad()
def validate_and_save_outputs(model: nn.Module):
    model.eval()

    scenes = []
    for s in sorted(os.listdir(VAL_MULTI_ROOT)):
        d = os.path.join(VAL_MULTI_ROOT, s)
        if not os.path.isdir(d):
            continue
        imgs = sorted(glob.glob(os.path.join(d, "*.*")))
        if len(imgs) >= 2:
            scenes.append(s)

    rows = []

    for i, scene in enumerate(scenes, start=1):
        try:
            in_dir = os.path.join(VAL_MULTI_ROOT, scene)
            gt_dir = os.path.join(VAL_GT_ROOT, scene)

            in_paths = sorted([p for p in glob.glob(os.path.join(in_dir, "*.*")) if os.path.isfile(p)])
            gt_path = pick_gt_image_for_scene(gt_dir)
            if gt_path is None or len(in_paths) == 0:
                continue

            in_imgs = [Image.open(p).convert("RGB") for p in in_paths]
            gt_img  = Image.open(gt_path).convert("RGB")

            hs = [im.size[1] for im in in_imgs] + [gt_img.size[1]]
            ws = [im.size[0] for im in in_imgs] + [gt_img.size[0]]
            h, w = min(hs), min(ws)

            in_imgs = [center_crop_pil(im, h, w) for im in in_imgs]
            inputs = [tf(im).unsqueeze(0).to(device) for im in in_imgs]

            pred = model(*inputs)
            fused_rgb_u8 = denorm_to_uint8(pred.squeeze(0))

            gt_rgb_u8 = np.array(Image.open(gt_path).convert("RGB"), dtype=np.uint8)
            fused_rgb_u8, gt_rgb_u8 = align_same_size_np(fused_rgb_u8, gt_rgb_u8)

            fused_g = to_gray_uint8(fused_rgb_u8)
            gt_g    = to_gray_uint8(gt_rgb_u8)

            viff_val  = vif_p_optional(fused_g, gt_g)
            issim_val = ssim_gray(fused_g, gt_g)
            psnr_val  = psnr_uint8(fused_rgb_u8, gt_rgb_u8)

            rows.append({
                "scene": scene,
                "gt_path": gt_path,
                "VIFF": viff_val,
                "ISSIM": issim_val,
                "PSNR": psnr_val
            })

            if i % 20 == 0 or i == len(scenes):
                print(f"[VAL] {i:04d}/{len(scenes)} done")

        except Exception as e:
            print(f"[VAL-SKIP] scene={scene} error: {e}")

    with open(CSV_METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "gt_path", "VIFF", "ISSIM", "PSNR"])
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) > 0:
        viff_m  = mean_ignore_nan([r["VIFF"] for r in rows])
        issim_m = mean_ignore_nan([r["ISSIM"] for r in rows])
        psnr_m  = mean_ignore_nan([r["PSNR"] for r in rows])

        print("\n=== VAL METRICS vs Ground Truth (model_conv1) ===")
        print(f"Scenes evaluated: {len(rows)}")
        print(f"Mean VIFF : {viff_m}")
        print(f"Mean ISSIM: {issim_m}")
        print(f"Mean PSNR : {psnr_m}")
        print(f"[VAL] CSV saved to: {CSV_METRICS_PATH}")
        return viff_m, issim_m, psnr_m, len(rows)


    if not _HAS_PIQ:
        print("\n[NOTE] VIFF is NaN because PIQ is not installed (optional).")
        print("       Install: pip install piq")

    return float("nan"), float("nan"), float("nan"), 0

# =========================================================
# 11) Main
# =========================================================
def main():
    train_ds = SceneDataset(TRAIN_MULTI_ROOT, TRAIN_GT_ROOT)
    val_ds   = SceneDataset(VAL_MULTI_ROOT, VAL_GT_ROOT)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_scene
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_scene
    )

    model = myIFCNN(fuse_scheme=0)

    base_sd = torch.load(BASELINE_CKPT, map_location="cpu")
    if isinstance(base_sd, dict) and "model_state_dict" in base_sd:
        base_sd = base_sd["model_state_dict"]


    missing, unexpected = model.load_state_dict(base_sd, strict=False)

    print("[LOAD] Loaded baseline (strict=False)")
    print("[LOAD] Missing keys:", missing)
    print("[LOAD] Unexpected keys:", unexpected)
    print("Missing:", len(missing))
    print("Unexpected:", len(unexpected))

    # ===============================
    # Train only conv layers (conv1,2,3), freeze BN
    # ===============================

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # for p in model.conv1.parameters():
    #     p.requires_grad = True

    # train conv2, conv3, conv4
    for p in model.conv2.parameters():
        p.requires_grad = True
    for p in model.conv3.parameters():
        p.requires_grad = True
    for p in model.conv4.parameters():
        p.requires_grad = True

    # freeze BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    model.to(device)

    perc_net = ResNet101Perceptual(RESNET101_CKPT).to(device)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        momentum=0.9,
        nesterov=True,
        weight_decay=WEIGHT_DECAY
    )



    import math

    updates_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    total_updates = EPOCHS * max(1, updates_per_epoch)
    cur_update = 0

    epochs_hist: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    viff_hist: List[float] = []
    issim_hist: List[float] = []
    psnr_hist: List[float] = []

    # init loss history CSV
    with open(LOSS_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()

    with open(METRICS_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "mean_VIFF", "mean_ISSIM", "mean_PSNR", "n_scenes"])
        w.writeheader()

    t0 = time.time()

    for ep in range(1, EPOCHS + 1):

        train_loss, cur_update = run_one_epoch_train(
            model, perc_net, train_loader, optimizer, cur_update, total_updates
        )

        val_loss = run_one_epoch_val_loss(model, perc_net, val_loader)

        epochs_hist.append(ep)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        print(
            f"\nEpoch {ep:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )


        with open(LOSS_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            w.writerow({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})


        plot_loss_curve(epochs_hist, train_losses, val_losses, LOSS_PLOT_PATH)


        viff_m, issim_m, psnr_m, n_scenes = validate_and_save_outputs(model)

        viff_hist.append(float(viff_m))
        issim_hist.append(float(issim_m))
        psnr_hist.append(float(psnr_m))

        with open(METRICS_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "mean_VIFF", "mean_ISSIM", "mean_PSNR", "n_scenes"])
            w.writerow({
                "epoch": ep,
                "mean_VIFF": viff_m,
                "mean_ISSIM": issim_m,
                "mean_PSNR": psnr_m,
                "n_scenes": n_scenes
            })

        plot_metrics_curve(epochs_hist, viff_hist, issim_hist, psnr_hist, METRICS_PLOT_PATH)


        if ep % SAVE_EVERY == 0:
            ckpt_path = f"{CKPT_PREFIX}_ep{ep}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")

    torch.save(model.state_dict(), CKPT_LAST)
    print(f"\n[CKPT] Saved last: {CKPT_LAST}")

    print("\nDone.")
    print(f"Total time: {(time.time() - t0)/60:.1f} min")
    print(f"Loss history CSV: {LOSS_HISTORY_CSV}")
    print(f"Loss plot: {LOSS_PLOT_PATH}")
    print(f"Metrics CSV (latest): {CSV_METRICS_PATH}")
    print(f"Fused images (latest): {SAVE_DIR}")


if __name__ == "__main__":
    main()
