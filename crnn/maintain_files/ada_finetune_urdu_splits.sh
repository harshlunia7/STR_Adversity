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
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Sending all the real dataset files to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/urdu_frac_split /ssd_scratch/cvit/rafaelgetto/finetune_data;

tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_10.tar.gz -C /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split;
# tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_25.tar.gz -C /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split;
# tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_50.tar.gz -C /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split;
# tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_75.tar.gz -C /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split;
echo "untar finished";

rm /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_*.tar.gz;
echo "Removed tar file from ssd_scratch"

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_crnn_outside_cvit/urdu/best_model_urdu_crnn.pth /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting Finetuning on urdu_text 10 %";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_10/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_10/val \
--datasetname outside_cvit_urdu_split --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_urdu_crnn.pth \
--lan urdu --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_urdu_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/urdu_text_splits/urdu_text_10 \
--adadelta \
--displayInterval 10 \
--valInterval 10 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

# echo "Starting Finetuning on urdu_text 25 %";
# python3 finetune.py \
# --trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_25/train/real/train \
# --valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_25/val \
# --datasetname outside_cvit_urdu_split --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_urdu_crnn.pth \
# --lan urdu --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_urdu_char_list.txt \
# --batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/urdu_text_splits/urdu_text_25 \
# --adadelta \
# --displayInterval 40 \
# --valInterval 40 \
# --manualSeed 1234 --random_sample --deal_with_lossnan;

# echo "Starting Finetuning on urdu_text 50 %";
# python3 finetune.py \
# --trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_50/train/real/train \
# --valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_50/val \
# --datasetname outside_cvit_urdu_split --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_urdu_crnn.pth \
# --lan urdu --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_urdu_char_list.txt \
# --batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/urdu_text_splits/urdu_text_50 \
# --adadelta \
# --displayInterval 40 \
# --valInterval 40 \
# --manualSeed 1234 --random_sample --deal_with_lossnan;

# echo "Starting Finetuning on urdu_text 75 %";
# python3 finetune.py \
# --trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_75/train/real/train \
# --valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_75/val \
# --datasetname outside_cvit_urdu_split --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_urdu_crnn.pth \
# --lan urdu --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_urdu_char_list.txt \
# --batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/urdu_text_splits/urdu_text_75 \
# --adadelta \
# --displayInterval 40 \
# --valInterval 40 \
# --manualSeed 1234 --random_sample --deal_with_lossnan;

echo "deactivate environment";
conda deactivate;