#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --mem-per-cpu=1G
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1

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
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/prabhu_sir_large_str_real_data/parseq_odia_real_large_st /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_parseq_finetuned_indicstr12/parseq_odia_indicstr_71_30.ckpt /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for Odia";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_odia_indicstr_71_30.ckpt \
trainer.gpus=1 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=odia \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/parseq_odia_real_large_st \
model.max_label_length=38 \
dataset=real;

echo "deactivate environment";
conda deactivate; 
