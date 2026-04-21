#!/bin/bash
#SBATCH --account=windfall
#SBATCH --partition=gpu_windfall
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --job-name=resume_yolo_wf
#SBATCH --output=logs/resume_wf_%j.out

module purge
module load pytorch/nvidia/22.12

cd "$(dirname "$0")/../.."

mkdir -p logs

pytorch ultra_entry.py detect train resume=True model=runs/detect/outputs/baseline_main_retry/weights/last.pt