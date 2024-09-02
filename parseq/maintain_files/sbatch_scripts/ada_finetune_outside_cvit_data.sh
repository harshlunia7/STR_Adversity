#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 20
#SBATCH --partition=long
#SBATCH --mem-per-cpu=1G
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
rsync -aP rafaelgetto@ada.iiit.ac.in:/home2/rafaelgetto/REAL_OUTSIDE_CVIT_DATA_STR /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/parseq_bengali_trained_92_55.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/parseq_hindi_trained_95_61.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/parseq_urdu_trained_80_48.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for mlt-17 Bengali";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_bengali_trained_92_55.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=bengali \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt17_bengali \
model.max_label_length=38 \
dataset=real;

echo "Starting finetuning for mlt-19 Bengali";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_bengali_trained_92_55.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=bengali \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_bengali \
model.max_label_length=38 \
dataset=real;

echo "Starting finetuning for mlt-19 Hindi";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_hindi_trained_95_61.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=hindi \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/mlt19_hindi \
model.max_label_length=50 \
dataset=real;

echo "Starting finetuning for Urdu";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=urdu \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/REAL_OUTSIDE_CVIT_DATA_STR/urdu_text \
model.max_label_length=42 \
dataset=real;

echo "deactivate environment";
conda deactivate; 
