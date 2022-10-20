#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=24GB
#SBATCH --mail-user=<your email>
#SBATCH --mail-type=FAIL,END
#SBATCH -o <add some filename>-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env name>

wandb disabled 
export TRANSFORMER_CACHE="/scratch/general/vast/<your uNID>/huggingface_cache"
python ... 