#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
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
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ajoy_sir_data_modified_lmdb /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_parseq_finetuned_indicstr12/parseq_malayalam_indicstr_68_81.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for urdu_text_10";
./finetune.py +checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_malayalam_indicstr_68_81.ckpt \
trainer.val_check_interval=1 \
+trainer.gpus=4 \
trainer.max_epochs=15 \
charset=malayalam \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/ajoy_sir_data_modified_lmdb \
model.max_label_length=54 \
dataset=real \
data.remove_whitespace=false \
data.normalize_unicode=false \
data.augment=false \
model.batch_size=128;

echo "deactivate environment";
conda deactivate; 
