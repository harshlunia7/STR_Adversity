#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh;

# activate conda environment 
conda activate parseq;
echo "conda environment activated"; 

echo "Creating ssd_scratch/cvit/rafaelgetto directory";
mkdir /ssd_scratch/cvit/rafaelgetto;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all the real dataset files to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_files/Odia_2 /ssd_scratch/cvit/rafaelgetto/finetune_data;


# echo "Creating ssd_scratch/cvit/rafaelgetto/sanjana_models directory";
# mkdir /ssd_scratch/cvit/rafaelgetto;
# mkdir /ssd_scratch/cvit/rafaelgetto/sanjana_models;

# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_kannada.tar.gz /ssd_scratch/cvit/rafaelgetto/sanjana_models;
# echo "Dataset Copied";
# tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_kannada.tar.gz -C /ssd_scratch/cvit/rafaelgetto/sanjana_models;
# echo "untar finished";

# rm /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_kannada.tar.gz;
# echo "Removed tar file from ssd_scratch"

echo "Starting Training";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/Odia_2/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/Odia_2/val \
--datasetname new_annotated --pretrained /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/odia/best_model_odia_crnn.pth \
--lan odia --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_odia_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/fine_tune_outputs/Odia_2 \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "deactivate environment";
conda deactivate;