./finetune_english.py +checkpoint_finetune=/ssd_scratch/cvit/rafaelgetto/checkpoint/parseq-english.pt trainer.val_check_interval=200 +trainer.gpus=4 trainer.max_epochs=15 charset=english data.root_dir=/ssd_scratch/cvit/rafaelgetto/dataset/coco_text_and_art dataset=real data.remove_whitespace=false data.normalize_unicode=false data.augment=false model.batch_size=128;