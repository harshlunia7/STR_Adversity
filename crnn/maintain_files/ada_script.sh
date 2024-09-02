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

echo "Creating ssd_scratch/cvit/rafaelgetto/sanjana_models directory";
mkdir /ssd_scratch/cvit/rafaelgetto;
mkdir /ssd_scratch/cvit/rafaelgetto/sanjana_models;

rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_meitei_manipuri.tar.gz /ssd_scratch/cvit/rafaelgetto/sanjana_models;
echo "Dataset Copied";
tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_meitei_manipuri.tar.gz -C /ssd_scratch/cvit/rafaelgetto/sanjana_models;
echo "untar finished";

rm /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_meitei_manipuri.tar.gz;
echo "Removed tar file from ssd_scratch"

echo "Starting Training";
python3 mytrain.py --trainRoot /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_meitei_manipuri/train/synth/train \
--valRoot /ssd_scratch/cvit/rafaelgetto/sanjana_models/parseq_meitei_manipuri/val \
--arch crnn --lan meitei_manipuri --charlist /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/charlist/crnn_meitei_manipuri_char_list.txt \
--batchSize 32 --nepoch 15 --cuda --expr_dir /home2/rafaelgetto/sanjana_model/code/3/crnn_new/new_files/outputs/meitei_manipuri \
--displayInterval 30000 --valInterval 30000 --adadelta \
--manualSeed 1234 --random_sample --deal_with_lossnan;

echo "deactivate environment";
conda deactivate;
