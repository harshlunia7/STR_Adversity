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
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_results;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_data;
mkdir /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Sending all the real dataset files to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_files/Odia_2 /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/parseq_odia_trained_93_27.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for Odia_2";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_odia_trained_93_27.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=odia \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Odia_2 \
model.max_label_length=38 \
dataset=real;

echo "deactivate environment";
conda deactivate; 
