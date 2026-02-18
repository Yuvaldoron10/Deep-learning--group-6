# coding: utf-8
"""
TEST SCRIPT - ORIGINAL IFCNN

"""

import os
import glob
import csv
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

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
print("piq available:", _HAS_PIQ)


# =========================================================
# 1) Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(SCRIPT_DIR, "data-image-splits")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

TEST_MULTI_ROOT = os.path.join(TEST_ROOT, "multi focus")
TEST_GT_ROOT    = os.path.join(TEST_ROOT, "Ground_truth")

CKPT_PATH = os.path.join(SCRIPT_DIR, "snapshots/IFCNN-MAX-original_ep20.pth")

assert os.path.isdir(TEST_MULTI_ROOT), f"Missing: {TEST_MULTI_ROOT}"
assert os.path.isdir(TEST_GT_ROOT), f"Missing: {TEST_GT_ROOT}"
assert os.path.isfile(CKPT_PATH), f"Missing checkpoint: {CKPT_PATH}"

OUT_DIR = os.path.join(SCRIPT_DIR, "results_test", "results_original")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "metrics_test.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.txt")


# =========================================================
# 2) Transforms (ImageNet normalize) + denorm for saving
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
# 3) Helpers: crop, GT picking (recursive), metrics
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
# 4) Load model + weights
# =========================================================
def load_checkpoint_into_model(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] checkpoint: {ckpt_path}")
    print(f"[LOAD] Missing keys: {missing}")
    print(f"[LOAD] Unexpected keys: {unexpected}")
    return missing, unexpected


# =========================================================
# 5) Run TEST
# =========================================================
@torch.no_grad()
def run_test():
    model = myIFCNN(fuse_scheme=0)
    load_checkpoint_into_model(model, CKPT_PATH)
    model.to(device)
    model.eval()


    scenes = []
    for s in sorted(os.listdir(TEST_MULTI_ROOT)):
        d = os.path.join(TEST_MULTI_ROOT, s)
        if not os.path.isdir(d):
            continue
        imgs = sorted(glob.glob(os.path.join(d, "*.*")))
        if len(imgs) >= 2:
            scenes.append(s)

    print(f"[TEST] scenes found: {len(scenes)}")

    rows = []
    for i, scene in enumerate(scenes, start=1):
        in_dir = os.path.join(TEST_MULTI_ROOT, scene)
        gt_dir = os.path.join(TEST_GT_ROOT, scene)

        in_paths = sorted([p for p in glob.glob(os.path.join(in_dir, "*.*")) if os.path.isfile(p)])
        gt_path = pick_gt_image_for_scene(gt_dir)
        if gt_path is None or len(in_paths) == 0:
            print(f"[SKIP] scene={scene} missing inputs/gt")
            continue


        in_imgs = [Image.open(p).convert("RGB") for p in in_paths]
        gt_img  = Image.open(gt_path).convert("RGB")

        hs = [im.size[1] for im in in_imgs] + [gt_img.size[1]]
        ws = [im.size[0] for im in in_imgs] + [gt_img.size[0]]
        h, w = min(hs), min(ws)

        in_imgs = [center_crop_pil(im, h, w) for im in in_imgs]
        gt_img  = center_crop_pil(gt_img, h, w)

        inputs = [tf(im).unsqueeze(0).to(device) for im in in_imgs]  # (1,3,H,W)
        pred = model(*inputs)  # (1,3,H,W)

        fused_u8 = denorm_to_uint8(pred.squeeze(0))
        gt_u8 = np.array(gt_img, dtype=np.uint8)

        fused_u8, gt_u8 = align_same_size_np(fused_u8, gt_u8)
        fused_g = to_gray_uint8(fused_u8)
        gt_g    = to_gray_uint8(gt_u8)

        viff_val  = vif_p_optional(fused_g, gt_g)
        issim_val = ssim_gray(fused_g, gt_g)
        psnr_val  = psnr_uint8(fused_u8, gt_u8)

        # save fused image
        out_path = os.path.join(OUT_DIR, f"{scene}.png")
        Image.fromarray(fused_u8).save(out_path)

        rows.append({
            "scene": scene,
            "gt_path": gt_path,
            "out_path": out_path,
            "VIFF": viff_val,
            "ISSIM": issim_val,
            "PSNR": psnr_val
        })

        if i % 20 == 0 or i == len(scenes):
            print(f"[TEST] {i:04d}/{len(scenes)} done")


    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene","gt_path","out_path","VIFF","ISSIM","PSNR"])
        writer.writeheader()
        writer.writerows(rows)

    viff_m  = mean_ignore_nan([r["VIFF"] for r in rows])
    issim_m = mean_ignore_nan([r["ISSIM"] for r in rows])
    psnr_m  = mean_ignore_nan([r["PSNR"] for r in rows])

    summary = (
        f"Checkpoint: {CKPT_PATH}\n"
        f"Scenes evaluated: {len(rows)}\n"
        f"Mean VIFF : {viff_m}\n"
        f"Mean ISSIM: {issim_m}\n"
        f"Mean PSNR : {psnr_m}\n"
        f"CSV: {CSV_PATH}\n"
        f"Outputs dir: {OUT_DIR}\n"
    )
    print("\n=== TEST METRICS vs Ground Truth (ORIGINAL) ===")
    print(summary)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    run_test()
