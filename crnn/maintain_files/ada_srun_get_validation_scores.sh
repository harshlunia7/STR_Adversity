rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/parseq_assamese.tar.gz /ssd_scratch/cvit/rafaelgetto;
tar -xvzf /ssd_scratch/cvit/rafaelgetto/parseq_assamese.tar.gz -C /ssd_scratch/cvit/rafaelgetto

python3 demo_org_cuda.py \
--model ./new_files/outputs/assamese/best_model_assamese_crnn.pth \
--val_result_dir ./new_files/outputs/Validation_txt_files \
--lan assamese \
--test_data /ssd_scratch/cvit/rafaelgetto/parseq_assamese/val \
--lexicon ./new_files/charlist/crnn_assamese_char_list.txt \
--type crnn;
