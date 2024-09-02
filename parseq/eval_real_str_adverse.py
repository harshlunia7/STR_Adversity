
import argparse

import torch

from PIL import Image
import os
import re

from pathlib import Path
from pathlib import PurePath
import yaml
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.models.parseq.system import PARSeq

def _get_config_english(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parent
    with open(root / 'configs/english.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config

def calculate_crr_wrr(groundtruth_texts, predicted_texts):
    """
    Calculate Character Recognition Rate (CRR) and Word Recognition Rate (WRR).

    :param groundtruth_texts: List of ground truth texts.
    :param predicted_texts: List of predicted texts.
    :return: Tuple containing CRR and WRR.
    """
    
    # Initialize counters
    total_characters = 0
    correctly_recognized_characters = 0
    correctly_recognized_words = 0

    # Iterate over the ground truth and predicted texts
    for gt_text, pred_text in zip(groundtruth_texts, predicted_texts):
        # Calculate the number of correctly recognized characters
        correctly_recognized_characters += sum(1 for gt_char, pred_char in zip(gt_text, pred_text) if gt_char == pred_char)
        # Add the length of the ground truth text to the total characters
        total_characters += len(gt_text)
        
        # Check if the entire word is correctly recognized
        if gt_text == pred_text:
            correctly_recognized_words += 1

    # Calculate CRR as the ratio of correctly recognized characters to total characters
    crr = correctly_recognized_characters / total_characters if total_characters > 0 else 0

    # Calculate WRR as the ratio of correctly recognized words to total words
    wrr = correctly_recognized_words / len(groundtruth_texts) if groundtruth_texts else 0

    return crr, wrr

def clean_text(text):
    # Remove all characters other than lowercase, uppercase, and English numerals
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--input', help='Images to read')
    parser.add_argument('--gt_file', help='gt file to check')
    parser.add_argument('--device', default='cuda')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    # model = PARSeq(**(_get_config_english("parseq", **kwargs)))
    # model.load_state_dict(torch.load(args.checkpoint))
    # model = model.eval().to(args.device)

    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    output_file = os.path.join(args.input, "evaluated_parseq_crr_wrr.txt")

    for dataset_dir in os.listdir(args.input):
        dataset_path = os.path.join(args.input, dataset_dir)
        print(dataset_dir, dataset_path)


        if os.path.isdir(dataset_path):
            processed_text_file = os.path.join(dataset_path, os.path.basename(args.gt_file))
            print(processed_text_file)

            # Find the processed_for_eval.txt file in the current dataset directory
            # for file_name in os.listdir(dataset_path):
            #     # if file_name.endswith('processed_for_eval.txt'):
            #     if file_name == os.path.basename(args.gt_file):
            #         processed_text_file = os.path.join(dataset_path, file_name)
            #         break

            if os.path.isfile(processed_text_file):
                groundtruth_texts = []
                predicted_texts = []

                with open(processed_text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                count_records_processed = 0
                for line in lines:
                    if count_records_processed % 100 == 0:
                        print(f"Number of records processed: {count_records_processed}")

                    image_filename, groundtruth_text = line.strip().split('\t')
                    image_path = os.path.join(dataset_path, image_filename)

                    image = Image.open(image_path).convert('RGB')
                    image = img_transform(image).unsqueeze(0).to(args.device)

                    p = model(image).softmax(-1)
                    pred, p = model.tokenizer.decode(p)

                    # Store ground truth and predicted texts
                    groundtruth_texts.append(groundtruth_text)
                    predicted_texts.append(pred[0])
                    count_records_processed += 1

                # Calculate CRR and WRR
                crr, wrr = calculate_crr_wrr(groundtruth_texts, predicted_texts)

                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"Dataset: {dataset_dir}\n")
                    out_f.write(f"\tCharacter Recognition Rate (CRR): {crr:.4f}\n")
                    out_f.write(f"\tWord Recognition Rate (WRR): {wrr:.4f}\n")
                print(f"Total Samples {len(lines)}")
                print(f"Processed dataset: {dataset_dir} - CRR: {(crr)*100:.2f}, WRR: {(wrr)*100:.2f}")
            else:
                print(f"No processed_for_eval.txt file found in {dataset_dir}") 

    
if __name__ == '__main__':
    main()