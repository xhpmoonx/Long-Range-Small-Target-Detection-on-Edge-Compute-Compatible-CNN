from pathlib import Path
from PIL import Image
import shutil

# =============================
# Paths
# =============================
project_root = Path(__file__).resolve().parent.parent
img_root = project_root / "data" / "raw" / "file1" / "Anti-UAV-Tracking-V0"
gt_root = project_root / "data" / "raw" / "file2" / "Anti-UAV-Tracking-V0GT"
out_root = project_root / "data" / "dataset"

# =============================
# Output folders
# =============================
for split in ["train", "val"]:
    (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# =============================
# Simple video-level split
# last 4 videos -> validation
# =============================
val_videos = {"video17", "video18", "video19", "video20"}


def clamp(v, low, high):
    return max(low, min(v, high))


total_images = 0
total_labels = 0
warnings = []

video_dirs = sorted([p for p in img_root.iterdir() if p.is_dir()])

for video_dir in video_dirs:
    video_name = video_dir.name
    split = "val" if video_name in val_videos else "train"

    gt_file = gt_root / f"{video_name}_gt.txt"
    if not gt_file.exists():
        warnings.append(f"[WARN] Missing GT file for {video_name}")
        continue

    image_files = sorted(video_dir.glob("*.jpg"))
    gt_lines = [line.strip() for line in gt_file.read_text().splitlines() if line.strip()]

    if len(image_files) != len(gt_lines):
        warnings.append(
            f"[WARN] {video_name}: {len(image_files)} images vs {len(gt_lines)} GT lines. "
            f"Using min={min(len(image_files), len(gt_lines))}"
        )

    n = min(len(image_files), len(gt_lines))

    for i in range(n):
        img_path = image_files[i]
        gt = gt_lines[i].split()

        if len(gt) != 4:
            warnings.append(f"[WARN] {video_name} line {i+1}: expected 4 values, got {len(gt)}")
            continue

        try:
            x, y, w, h = map(float, gt)
        except ValueError:
            warnings.append(f"[WARN] {video_name} line {i+1}: non-numeric GT")
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        stem = img_path.stem
        new_name = f"{video_name}_{stem}"

        out_img = out_root / "images" / split / f"{new_name}.jpg"
        out_lbl = out_root / "labels" / split / f"{new_name}.txt"

        shutil.copy2(img_path, out_img)

        if w <= 0 or h <= 0:
            warnings.append(
                f"[WARN] {video_name} line {i+1}: non-positive bbox -> saved as background frame"
            )
            out_lbl.write_text("")
            total_images += 1
            total_labels += 1
            continue

        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        x_center = clamp(x_center, 0.0, 1.0)
        y_center = clamp(y_center, 0.0, 1.0)
        w_norm = clamp(w_norm, 0.0, 1.0)
        h_norm = clamp(h_norm, 0.0, 1.0)

        out_lbl.write_text(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        total_images += 1
        total_labels += 1

print("Done.")
print(f"Total images copied: {total_images}")
print(f"Total labels written: {total_labels}")

if warnings:
    print("\nWarnings:")
    for w in warnings[:50]:
        print(w)
    if len(warnings) > 50:
        print(f"... and {len(warnings) - 50} more warnings")