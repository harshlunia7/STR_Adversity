#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 20
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh; 

# activate conda environment 
conda activate parseq;
echo "conda environment activated";

echo "Creating ssd_scratch/cvit/rafaelgetto directory";
mkdir /ssd_scratch/cvit/rafaelgetto;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_results;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_data;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Sending all the real dataset files to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/multi_lang/hindi_gujarati /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_multi_lang/hindi_gujarati/multi_lang_hindi_gujarati_2M_parseq.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for hindi_gujarati_new_val_gujarati_new";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/multi_lang_hindi_gujarati_2M_parseq.ckpt \
trainer.gpus=2 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=hindi_gujarati \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/hindi_gujarati/hindi_gujarati_new_val_gujarati_new \
model.max_label_length=50 \
dataset=real;

echo "Starting finetuning for hindi_gujarati_new_val_hindi";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/multi_lang_hindi_gujarati_2M_parseq.ckpt \
trainer.gpus=2 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=hindi_gujarati \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/hindi_gujarati/hindi_gujarati_new_val_hindi \
model.max_label_length=50 \
dataset=real;

echo "Starting finetuning for hindi_gujarati_sanjana_val_gujarati_sanjana";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/multi_lang_hindi_gujarati_2M_parseq.ckpt \
trainer.gpus=2 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=hindi_gujarati \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/hindi_gujarati/hindi_gujarati_sanjana_val_gujarati_sanjana \
model.max_label_length=50 \
dataset=real;

echo "Starting finetuning for hindi_gujarati_sanjana_val_hindi";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/multi_lang_hindi_gujarati_2M_parseq.ckpt \
trainer.gpus=2 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=hindi_gujarati \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/hindi_gujarati/hindi_gujarati_sanjana_val_hindi \
model.max_label_length=50 \
dataset=real;

echo "deactivate environment";
conda deactivate; 
