#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 20
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh;

# activate conda environment 
conda activate parseq;
echo "conda environment activated"; 

echo "Creating ssd_scratch/cvit/rafaelgetto/crnn directory";
mkdir /ssd_scratch/cvit/rafaelgetto;
mkdir /ssd_scratch/cvit/rafaelgetto/crnn;
mkdir /ssd_scratch/cvit/rafaelgetto/crnn/finetuned_model;

rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/lmbd_data_gujarati /ssd_scratch/cvit/rafaelgetto/crnn;
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_models_trained_on_synth/crnn/gujarati /ssd_scratch/cvit/rafaelgetto/crnn/finetuned_model;
echo "Dataset Copied";

echo "Starting Finetuning";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/crnn/lmbd_data_gujarati/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/crnn/lmbd_data_gujarati/val \
--datasetname icpr_data --pretrained /ssd_scratch/cvit/rafaelgetto/crnn/finetuned_model/gujarati/best_model_gujarati_crnn.pth \
--lan gujarati --charlist /home2/rafaelgetto/crnn/new_files/charlist/crnn_gujarati_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/crnn/new_files/outputs/icpr_comp_models/gujarati \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "deactivate environment";
conda deactivate;
