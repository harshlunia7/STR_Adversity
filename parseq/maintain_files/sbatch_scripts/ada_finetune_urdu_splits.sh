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

rm /ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_*.tar.gz;
echo "Removed tar file from ssd_scratch";

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/parseq_urdu_trained_80_48.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for urdu_text_10";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
trainer.gpus=1 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=urdu \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_10 \
model.max_label_length=42 \
dataset=real;

# echo "Starting finetuning for urdu_text_25";
# ./finetune.py \
# +checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
# trainer.gpus=1 \
# trainer.val_check_interval=1 \
# trainer.max_epochs=15 \
# charset=urdu \
# data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_25 \
# model.max_label_length=42 \
# dataset=real;
# 
# echo "Starting finetuning for urdu_text_50";
# ./finetune.py \
# +checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
# trainer.gpus=1 \
# trainer.val_check_interval=1 \
# trainer.max_epochs=15 \
# charset=urdu \
# data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_50 \
# model.max_label_length=42 \
# dataset=real;

# echo "Starting finetuning for urdu_text_75";
# ./finetune.py \
# +checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
# trainer.gpus=1 \
# trainer.val_check_interval=1 \
# trainer.max_epochs=15 \
# charset=urdu \
# data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/urdu_frac_split/urdu_text_75 \
# model.max_label_length=42 \
# dataset=real;

echo "deactivate environment";
conda deactivate; 
