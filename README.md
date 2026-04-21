# Long-Range Small Target Detection on Edge Compute Compatible CNN

This project focuses on detecting small unmanned aerial vehicles (UAVs) at medium to long range using a lightweight YOLO-based detector suitable for edge-compatible deployment.

The current baseline is based on **YOLOv8 Nano**, trained and evaluated on the **DUT-Anti-UAV** dataset. The project is designed to support future architectural modifications such as early-layer changes, multi-scale head tuning, and attention modules for improved small-target detection.

---

## Project Goal

The main objective of this project is to build an efficient object detection pipeline for UAV detection in cluttered backgrounds while keeping the model lightweight enough for edge compute settings.

We are especially interested in:

- improving **small-target feature extraction**
- increasing **recall**
- improving **bounding box localization**
- preserving a lightweight architecture suitable for deployment

---

## Current Baseline Results

Baseline model: **YOLOv8 Nano**

Validation performance:

- **Precision:** 0.975
- **Recall:** 0.723
- **mAP50:** 0.767
- **mAP50-95:** 0.418

These results show that the baseline model is highly precise, meaning its positive predictions are usually correct, but recall remains lower than desired, indicating that some drones are still being missed. The gap between mAP50 and mAP50-95 also suggests that localization quality can be improved.

---

## Repository Structure

```text
Long-Range-Small-Target-Detection-on-Edge-Compute-Compatible-CNN/
├── configs/
│   └── drone.yaml
├── hpc/
│   └── run_resume_windfall.sh
├── scripts/
│   ├── prepare_yolo_dataset.py
│   └── ultra_entry.py
├── .gitignore
└── README.md
```

---

## Files

### `configs/drone.yaml`
YOLO dataset configuration file.

### `scripts/prepare_yolo_dataset.py`
Prepares the DUT-Anti-UAV raw data into YOLO-format train/validation image-label folders.

### `scripts/ultra_entry.py`
Helper entry script for running Ultralytics commands in environments where direct CLI execution is inconvenient, especially on HPC.

### `hpc/run_resume_windfall.sh`
Slurm batch script used to resume training on UArizona HPC using the `gpu_windfall` partition.

---

## Dataset

This project uses the **DUT-Anti-UAV** dataset.

The dataset itself is **not included** in this repository.

Expected local/HPC structure:

```text
data/
├── raw/
│   ├── file1/
│   │   └── Anti-UAV-Tracking-V0/
│   └── file2/
│       └── Anti-UAV-Tracking-V0GT/
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## Preparing the Dataset

After placing the raw dataset in the expected `data/raw/` structure, run:

```bash
python scripts/prepare_yolo_dataset.py
```

This creates the YOLO-format dataset under:

```text
data/dataset/
```

---

## Training

### Local test run

A quick local test can be run with:

```bash
python scripts/ultra_entry.py detect train model=yolov8n.pt data=configs/drone.yaml epochs=1 imgsz=640 batch=4 project=runs/detect/outputs name=temp_test
```

### HPC training

On UArizona HPC, training can be submitted through Slurm:

```bash
sbatch hpc/run_resume_windfall.sh
```

This script is intended for HPC use only.

---

## Outputs

Typical training outputs include:

- `best.pt`
- `last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- precision / recall / F1 / PR curves

These generated outputs are excluded from Git tracking.

---

## Future Work

Planned improvements include:

- early convolutional layer modification for better small-target feature extraction
- multi-scale head tuning to reduce emphasis on large objects
- attention module integration (such as CBAM or SE)
- augmentation and hyperparameter tuning
- comparison against the YOLOv8 Nano baseline

---

## Reproducibility Notes

This repository tracks:

- code
- configs
- helper scripts
- HPC batch scripts

It does **not** track:

- raw dataset files
- processed dataset files
- logs
- training outputs
- virtual environments
- pretrained or trained weight files
