# coding: utf-8

import os
import cv2
import time
import glob
import torch
from model import myIFCNN

from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

from utils.myTransforms import denorm
from utils.myDatasets import ImageSequence


# =========================
# 0) Device
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 1) Load IFCNN-MAX (multi-focus)
# =========================
model_name = 'IFCNN-MAX'
fuse_scheme = 0   # MAX fusion

model = myIFCNN(fuse_scheme=fuse_scheme)

ckpt = torch.load(os.path.join('snapshots', model_name + '.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt)

model.eval()
model = model.to(device)

# =========================
# 2) Paths (LOCAL data-image)
# =========================
# assumes data-image is next to this script / inside your project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(SCRIPT_DIR, "data-image")

multi_root = os.path.join(data_root, "multi focus")
gt_root    = os.path.join(data_root, "Ground_truth")

# sanity checks
if not os.path.isdir(data_root):
    raise FileNotFoundError(f"data_root not found: {data_root}")
if not os.path.isdir(multi_root):
    raise FileNotFoundError(f"'multi focus' folder not found: {multi_root}")
if not os.path.isdir(gt_root):
    raise FileNotFoundError(f"'Ground_truth' folder not found: {gt_root}")

# =========================
# 3) Run on YOUR dataset
# =========================
dataset = "MY_MF"
is_save = True
is_gray = False
is_folder = False

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

os.makedirs("results", exist_ok=True)

begin_time = time.time()

scenes = sorted([d for d in os.listdir(multi_root) if os.path.isdir(os.path.join(multi_root, d))])

for scene in scenes:
    scene_dir = os.path.join(multi_root, scene)

    img_paths = sorted(glob.glob(os.path.join(scene_dir, "*.*")))
    if len(img_paths) < 2:
        print(f"[SKIP] scene={scene} : found <2 images in {scene_dir}")
        continue

    seq_loader = ImageSequence(
        is_folder,
        "RGB",
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        *img_paths
    )

    imgs = seq_loader.get_imseq()

    with torch.no_grad():
        vimgs = []
        for img in imgs:
            img = img.unsqueeze(0)  # add batch
            vimgs.append(Variable(img.to(device)))
        vres = model(*vimgs)

        res = denorm(mean, std, vres[0]).clamp(0, 1) * 255
        res_img = res.detach().cpu().numpy().astype("uint8")
        fused = res_img.transpose([1, 2, 0])  # HWC RGB

    filename = f"{model_name}-{dataset}-{scene}"
    out_path = os.path.join("results", filename + ".png")

    if is_gray:
        fused = cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
        Image.fromarray(fused).save(out_path, format="PNG", compress_level=0)
    else:
        Image.fromarray(fused).save(out_path, format="PNG", compress_level=0)

    gt_dir = os.path.join(gt_root, scene)
    gt_candidates = sorted(glob.glob(os.path.join(gt_dir, "*.*")))
    if len(gt_candidates) > 0:
        print(f"[OK] scene={scene} saved -> {out_path} | GT -> {gt_candidates[0]}")
    else:
        print(f"[OK] scene={scene} saved -> {out_path} | (no GT found in {gt_dir})")

proc_time = time.time() - begin_time
print(f"Total processing time of {dataset} dataset: {proc_time:.3}s")


######################################

# =========================
# 4) Metrics vs Ground Truth (VIFF, ISSIM, LPIPS, PSNR)
# =========================
# Add this below your existing code (after the fusion loop).
# Computes per-scene metrics and saves CSV + prints MEAN metrics.

import csv
import math

import numpy as np
import cv2
from PIL import Image

# ---------- helpers ----------
def to_gray_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 HWC -> Gray uint8 HW"""
    if img_rgb_uint8.ndim == 2:
        return img_rgb_uint8.astype(np.uint8)
    return cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)

def align_same_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop both images to the same (minH, minW) if sizes differ."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)

    def center_crop(img, h, w):
        H, W = img.shape[:2]
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return img[y0:y0+h, x0:x0+w]

    return center_crop(a, h, w), center_crop(b, h, w)

def ssim_gray(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """
    ISSIM (we compute SSIM on grayscale).
    Uses skimage if available; falls back to a simple global SSIM otherwise.
    """
    gray_a = gray_a.astype(np.float64)
    gray_b = gray_b.astype(np.float64)

    try:
        from skimage.metrics import structural_similarity as ssim
        val = ssim(gray_a, gray_b, data_range=255)
        return float(val)
    except Exception:
        # Simple SSIM fallback (global, not windowed)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu1 = gray_a.mean()
        mu2 = gray_b.mean()
        sigma1 = gray_a.var()
        sigma2 = gray_b.var()
        sigma12 = ((gray_a - mu1) * (gray_b - mu2)).mean()
        num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
        return float(num / (den + 1e-12))

def psnr_uint8(a_uint8: np.ndarray, b_uint8: np.ndarray) -> float:
    """
    PSNR on uint8 images (RGB or Gray). Higher is better.
    Uses MAX_I = 255.
    """
    a = a_uint8.astype(np.float64)
    b = b_uint8.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))

def viff_p_optional(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """
    Compute VIFF using PIQ's VIF-P (pixel-domain VIF).
    Returns NaN if PIQ isn't available.
    IMPORTANT: This is VIF-P (often used as "VIFF" in practice).
    """
    try:
        import torch
        import piq
        ta = torch.from_numpy(gray_a.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # NCHW in [0,1]
        tb = torch.from_numpy(gray_b.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            v = piq.vif_p(ta, tb, data_range=1.0)
        return float(v.item())
    except Exception as e:
        # Uncomment to debug:
        # print("[VIFF ERROR]", repr(e))
        return float("nan")

def lpips_optional(fused_rgb_uint8: np.ndarray, gt_rgb_uint8: np.ndarray) -> float:
    """
    Compute LPIPS (perceptual distance). Lower is better.
    Tries 'lpips' package (recommended). Returns NaN if unavailable.

    Expects RGB uint8 HWC; converts to torch tensors in [-1, 1] with shape NCHW.
    """
    try:
        import torch
        import lpips  # pip install lpips

        # NCHW float in [0,1]
        a = torch.from_numpy(fused_rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(gt_rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        # LPIPS expects [-1, 1]
        a = a * 2.0 - 1.0
        b = b * 2.0 - 1.0

        # default backbone used widely: 'alex'
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.eval()

        with torch.no_grad():
            d = loss_fn(a, b)
        return float(d.item())
    except Exception as e:
        # Uncomment to debug:
        # print("[LPIPS ERROR]", repr(e))
        return float("nan")

def mean_ignore_nan(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / (len(vals) + 1e-12)) if len(vals) else float("nan")

# =========================
# 4A) Re-run over saved fused images + GT to compute metrics
# =========================
results_dir = os.path.join(SCRIPT_DIR, "results")
csv_path = os.path.join(results_dir, "metrics_vs_gt.csv")

rows = []
for scene in scenes:
    fused_path = os.path.join(results_dir, f"{model_name}-{dataset}-{scene}.png")
    gt_dir = os.path.join(gt_root, scene)
    gt_candidates = sorted(glob.glob(os.path.join(gt_dir, "*.*")))

    if not os.path.isfile(fused_path):
        print(f"[METRICS-SKIP] scene={scene} fused not found: {fused_path}")
        continue
    if len(gt_candidates) == 0:
        print(f"[METRICS-SKIP] scene={scene} GT not found in: {gt_dir}")
        continue

    gt_path = gt_candidates[0]

    # read images (RGB)
    fused_rgb = np.array(Image.open(fused_path).convert("RGB"), dtype=np.uint8)
    gt_rgb    = np.array(Image.open(gt_path).convert("RGB"), dtype=np.uint8)

    # align size (if needed)
    fused_rgb, gt_rgb = align_same_size(fused_rgb, gt_rgb)

    # grayscale for VIFF/ISSIM
    fused_g = to_gray_uint8(fused_rgb)
    gt_g    = to_gray_uint8(gt_rgb)

    # metrics
    viff_val  = viff_p_optional(fused_g, gt_g)          # VIFF (VIF-P via PIQ) on grayscale
    issim_val = ssim_gray(fused_g, gt_g)                # ISSIM (SSIM) on grayscale
    psnr_val  = psnr_uint8(fused_rgb, gt_rgb)           # PSNR on RGB

    rows.append({
        "scene": scene,
        "fused_path": fused_path,
        "gt_path": gt_path,
        "VIFF": viff_val,
        "ISSIM": issim_val,
        "PSNR": psnr_val
    })

# save CSV
os.makedirs(results_dir, exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["scene", "fused_path", "gt_path", "VIFF", "ISSIM", "PSNR"]
    )
    writer.writeheader()
    writer.writerows(rows)

# print summary (mean over scenes)
if len(rows) == 0:
    print("[METRICS] No rows computed (missing fused or GT).")
else:
    viff_m  = mean_ignore_nan([r["VIFF"] for r in rows])
    issim_m = mean_ignore_nan([r["ISSIM"] for r in rows])
    psnr_m  = mean_ignore_nan([r["PSNR"] for r in rows])

    print("\n=== METRICS vs Ground Truth (per-scene saved to metrics_vs_gt.csv) ===")
    print(f"Scenes evaluated: {len(rows)}")
    print(f"Mean VIFF : {viff_m}")
    print(f"Mean ISSIM: {issim_m}")
    print(f"Mean PSNR : {psnr_m}")
    print(f"\n[METRICS] CSV saved to: {csv_path}")

    # Installation notes
    if all(isinstance(r["VIFF"], float) and math.isnan(r["VIFF"]) for r in rows):
        print("\n[NOTE] VIFF is NaN because PIQ (vif_p) is not available.")
        print("       Install and rerun: pip install piq")