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
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_files/* /ssd_scratch/cvit/rafaelgetto/finetune_data;

echo "Sending all checkpoints to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/ckp_finetune_parseq/* /ssd_scratch/cvit/rafaelgetto/finetune_checkpoints;

echo "Starting finetuning for Kannada";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_kannada_trained_81_18.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=kannadafinetune \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Kannada \
model.max_label_length=41 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Kannada;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Kannada rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Kannada;

echo "Starting finetuning for Malayalam";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_malayalam_trained_92_27.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=malayalam \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Malayalam \
model.max_label_length=54 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Malayalam;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Malayalam rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Malayalam;

echo "Starting finetuning for Marathi";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_marathi_trained_95_82.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=marathi \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Marathi \
model.max_label_length=45 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Marathi;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Marathi rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Marathi;

echo "Starting finetuning for Odia";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_odia_trained_93_27.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=odia \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Odia \
model.max_label_length=38 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Odia;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Odia rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Odia;

echo "Starting finetuning for Punjabi";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_punjabi_trained_87_89.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=punjabi \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Punjabi \
model.max_label_length=32 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Punjabi;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Punjabi rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Punjabi;

echo "Starting finetuning for Tamil";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_tamil_trained_90_31.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=tamil \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Tamil \
model.max_label_length=36 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Tamil;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Tamil rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Tamil;

echo "Starting finetuning for Telugu";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_telugu_trained_83_40.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=telugu \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Telugu \
model.max_label_length=51 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Telugu;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Telugu rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Telugu;


echo "Starting finetuning for Urdu";
./finetune.py \
+checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/finetune_checkpoints/parseq_urdu_trained_80_48.ckpt \
trainer.gpus=0 \
trainer.val_check_interval=1 \
trainer.max_epochs=15 \
charset=urdu \
data.root_dir=/ssd_scratch/cvit/rafaelgetto/finetune_data/Urdu \
model.max_label_length=42 \
dataset=real;

echo "Change output directory name in home2";
mv /home2/rafaelgetto/parseq/outputs/parseq/2* /home2/rafaelgetto/parseq/outputs/parseq/Urdu;

echo "Sending the output to share3";
rsync -aP /home2/rafaelgetto/parseq/outputs/parseq/Urdu rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/finetune_parseq_results;

echo "Remove output from home";
rm -rf /home2/rafaelgetto/parseq/outputs/parseq/Urdu;


echo "deactivate environment";
conda deactivate; 
