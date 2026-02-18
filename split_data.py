import shutil
import random
from pathlib import Path

# =========================================================
# CONFIG (percentage split)
# =========================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42

OVERWRITE_SPLITS = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================================================
# PATHS (relative to this script)
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_ROOT  = SCRIPT_DIR / "data-image"
MULTI_ROOT = DATA_ROOT / "multi focus"
GT_ROOT    = DATA_ROOT / "Ground_truth"

SPLIT_ROOT = SCRIPT_DIR / "data-image-splits"


# =========================================================
# Helpers
# =========================================================
def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_scene_ids(multi_root: Path):
    if not multi_root.is_dir():
        raise FileNotFoundError(f"'multi focus' folder not found: {multi_root}")
    return sorted([d.name for d in multi_root.iterdir() if d.is_dir()])


def ensure_empty_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Folder already exists: {path}\n"
                f"Set OVERWRITE_SPLITS=True to delete and recreate it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def scene_is_valid(scene: str) -> bool:
    src_multi = MULTI_ROOT / scene
    if not src_multi.is_dir():
        return False
    mf_imgs = [p for p in src_multi.iterdir() if is_image_file(p)]
    return len(mf_imgs) >= 2


def move_scene(scene: str, split_name: str) -> bool:
    src_multi = MULTI_ROOT / scene
    src_gt    = GT_ROOT / scene

    dst_multi = SPLIT_ROOT / split_name / "multi focus" / scene
    dst_gt    = SPLIT_ROOT / split_name / "Ground_truth" / scene

    if not src_multi.is_dir():
        print(f"[WARN] scene={scene}: missing multi focus dir -> SKIP")
        return False

    mf_imgs = [p for p in src_multi.iterdir() if is_image_file(p)]
    if len(mf_imgs) < 2:
        print(f"[WARN] scene={scene}: <2 images in multi focus -> SKIP")
        return False


    dst_multi.parent.mkdir(parents=True, exist_ok=True)
    dst_gt.parent.mkdir(parents=True, exist_ok=True)

    if dst_multi.exists():
        shutil.rmtree(dst_multi)
    shutil.move(str(src_multi), str(dst_multi))

    if src_gt.is_dir():
        if dst_gt.exists():
            shutil.rmtree(dst_gt)
        shutil.move(str(src_gt), str(dst_gt))
    else:
        print(f"[WARN] scene={scene}: missing Ground_truth folder (moved only multi focus).")

    return True


def make_split_folders():
    for split_name in ["train", "val", "test"]:
        (SPLIT_ROOT / split_name / "multi focus").mkdir(parents=True, exist_ok=True)
        (SPLIT_ROOT / split_name / "Ground_truth").mkdir(parents=True, exist_ok=True)


# =========================================================
# Main
# =========================================================
def main():
    print("=== Dataset Split (MOVE) | 70/15/15 by percentage ===")
    print(f"[INFO] Script dir : {SCRIPT_DIR}")
    print(f"[INFO] DATA_ROOT  : {DATA_ROOT}")
    print(f"[INFO] MULTI_ROOT : {MULTI_ROOT}")
    print(f"[INFO] GT_ROOT    : {GT_ROOT}")
    print(f"[INFO] SPLIT_ROOT : {SPLIT_ROOT}\n")

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"data-image folder not found next to script: {DATA_ROOT}")
    if not MULTI_ROOT.is_dir():
        raise FileNotFoundError(f"'multi focus' folder not found: {MULTI_ROOT}")
    if not GT_ROOT.is_dir():
        raise FileNotFoundError(f"'Ground_truth' folder not found: {GT_ROOT}")

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0. Got {total_ratio}")

    random.seed(SEED)

    all_scenes = list_scene_ids(MULTI_ROOT)
    print(f"[INFO] Found {len(all_scenes)} scene folders in multi focus.")

    valid_scenes = [s for s in all_scenes if scene_is_valid(s)]
    n = len(valid_scenes)
    print(f"[INFO] Valid scenes (>=2 MF images): {n}")

    if n == 0:
        raise ValueError("No valid scenes found (need >=2 images per scene in multi focus).")

    random.shuffle(valid_scenes)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_scenes = valid_scenes[:n_train]
    val_scenes   = valid_scenes[n_train:n_train + n_val]
    test_scenes  = valid_scenes[n_train + n_val:]

    print(f"[INFO] Split sizes computed: train={len(train_scenes)}, val={len(val_scenes)}, test={len(test_scenes)}")
    print(f"[INFO] Ratios achieved: train={len(train_scenes)/n:.3f}, val={len(val_scenes)/n:.3f}, test={len(test_scenes)/n:.3f}")


    ensure_empty_dir(SPLIT_ROOT, overwrite=OVERWRITE_SPLITS)
    make_split_folders()

    moved = {"train": 0, "val": 0, "test": 0}

    for s in train_scenes:
        if move_scene(s, "train"):
            moved["train"] += 1

    for s in val_scenes:
        if move_scene(s, "val"):
            moved["val"] += 1

    for s in test_scenes:
        if move_scene(s, "test"):
            moved["test"] += 1

    print("\n=== DONE ===")
    print("Moved scenes:")
    for k, v in moved.items():
        print(f"  {k}: {v}")

    print(f"\n[OK] Output created at:\n  {SPLIT_ROOT}")


    leftovers = []
    for d in MULTI_ROOT.iterdir():
        if d.is_dir():
            leftovers.append(d.name)
    if leftovers:
        print("\n[NOTE] Some folders remained in original multi focus (usually invalid scenes or non-scene folders).")
        print(f"Remaining folders count: {len(leftovers)}")
        print("Example:", leftovers[:10])


if __name__ == "__main__":
    main()