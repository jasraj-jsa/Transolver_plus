#!/bin/bash
#SBATCH --job-name="transolver"
#SBATCH --output=transolver_%j.out
#SBATCH --error=transolver_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait

# Load HPC modules FIRST (before venv activation)
# module load 2025
# module load py-torch/2.5.1

# Activate your venv
# source /home/jsanand/Transolver_plus/.venv/bin/activate
# source activate transolver

# Run the training script
# srun bash scripts/transolver_plus.sh

# echo "which python: $(which python)"
# echo "python path: $(python -c 'import sys; print(sys.executable)')"
# echo "torch check: $(python -c 'import torch; print(torch.__file__)' 2>&1)"
# echo "vtk check: $(python -c 'import vtk; print(vtk.__file__)' 2>&1)"
module load 2025
module load python
module load py-torch/2.5.1

rm /home/jsanand/Transolver_plus/.venv/bin/python* 2>/dev/null
ln -s $(which python) /home/jsanand/.venv/bin/python
ln -s $(which python3) /home/jsanand/.venv/bin/python3

source /home/jsanand/.venv/bin/activate


# working directory
cd /home/jsanand/Transolver_plus

export CUDA_VISIBLE_DEVICES=0

python main_car.py \
  --data_dir ./dataset/mlcfd_data/car_preprocessed \
  --save_dir ./dataset/mlcfd_data/car_preprocessed \
  --preprocessed 1 \
  --fold_id 0 --gpu 0 --nb_epochs 200 --lr 0.001