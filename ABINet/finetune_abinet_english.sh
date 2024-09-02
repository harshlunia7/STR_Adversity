#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 20
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh;

# activate conda environment 
conda activate abinet;
echo "conda environment activated"; 

echo "Creating ssd_scratch/cvit/rafaelgetto/dataset directory";
mkdir /ssd_scratch/cvit/rafaelgetto;
mkdir /ssd_scratch/cvit/rafaelgetto/dataset;
mkdir /ssd_scratch/cvit/rafaelgetto/dataset/non_degraded;
# mkdir /ssd_scratch/cvit/rafaelgetto/dataset/degraded;
# mkdir /ssd_scratch/cvit/rafaelgetto/dataset/degraded/fog;
# mkdir /ssd_scratch/cvit/rafaelgetto/dataset/degraded/rain;
# mkdir /ssd_scratch/cvit/rafaelgetto/dataset/degraded/snow;
# mkdir /ssd_scratch/cvit/rafaelgetto/dataset/degraded/lowlight;

rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/cvip_str_adverse/datasets/non_degraded/coco_text_and_art /ssd_scratch/cvit/rafaelgetto/dataset/non_degraded;
# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/cvip_str_adverse/datasets/degraded/fog/coco_text_and_art /ssd_scratch/cvit/rafaelgetto/dataset/degraded/fog;
# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/cvip_str_adverse/datasets/degraded/snow/coco_text_and_art /ssd_scratch/cvit/rafaelgetto/dataset/degraded/snow;
# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/cvip_str_adverse/datasets/degraded/rain/coco_text_and_art /ssd_scratch/cvit/rafaelgetto/dataset/degraded/rain;
# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/cvip_str_adverse/datasets/degraded/lowlight/coco_text_and_art /ssd_scratch/cvit/rafaelgetto/dataset/degraded/lowlight;
echo "Dataset Copied";

echo "Starting Finetuning on nondegraded data";
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet_non_degraded.yaml

# echo "Starting Finetuning on fog degraded data";
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet_fog_degraded.yaml

# echo "Starting Finetuning on snow degraded data";
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet_snow_degraded.yaml

# echo "Starting Finetuning on rain degraded data";
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet_rain_degraded.yaml

# echo "Starting Finetuning on lowlight degraded data";
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet_lowlight_degraded.yaml

echo "deactivate environment";
conda deactivate;