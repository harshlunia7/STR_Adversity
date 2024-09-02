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
rsync -aP rafaelgetto@ada.iiit.ac.in:/home2/rafaelgetto/REAL_OUTSIDE_CVIT_DATA_STR /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_crnn_outside_cvit/*/*crnn.pth /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;


echo "Starting Finetuning on mlt17_bengali";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt17_bengali/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt17_bengali/val \
--datasetname outside_cvit --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_bengali_crnn.pth \
--lan bengali --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_bengali_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/finetuned_outside_cvit/mlt17_bengali \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "Starting Finetuning on mlt19_bengali";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_bengali/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_bengali/val \
--datasetname outside_cvit --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_bengali_crnn.pth \
--lan bengali --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_bengali_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/finetuned_outside_cvit/mlt19_bengali \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "Starting Finetuning on mlt19_hindi";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_hindi/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_hindi/val \
--datasetname outside_cvit --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_hindi_crnn.pth \
--lan hindi --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_hindi_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/finetuned_outside_cvit/mlt19_hindi \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "Starting Finetuning on urdu_text";
python3 finetune.py \
--trainRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/urdu_text/train/real/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/urdu_text/val \
--datasetname outside_cvit --pretrained /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/best_model_urdu_crnn.pth \
--lan urdu --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_urdu_char_list.txt \
--batchSize 32 --nepoch 50 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/finetuned_outside_cvit/urdu_text \
--adadelta \
--displayInterval 40 \
--valInterval 40 \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "deactivate environment";
conda deactivate;