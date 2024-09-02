
import argparse

import torch

from PIL import Image
import os

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--input', help='Images to read')
    parser.add_argument('--device', default='cuda')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    output_file = os.path.join(args.input, "evaluated_parseq_crr_wrr.txt")

    for dataset_dir in os.listdir(args.input):
        dataset_path = os.path.join(args.input, dataset_dir)

        if os.path.isdir(dataset_path):
            processed_text_file = None
            # Find the processed_for_eval.txt file in the current dataset directory
            for file_name in os.listdir(dataset_path):
                if file_name.endswith('processed_for_eval.txt'):
                    processed_text_file = os.path.join(dataset_path, file_name)
                    break

            if processed_text_file:
                groundtruth_texts = []
                predicted_texts = []

                with open(processed_text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    image_filename, groundtruth_text = line.strip().split('\t')
                    image_path = os.path.join(dataset_path, image_filename)

                    image = Image.open(image_path).convert('RGB')
                    image = img_transform(image).unsqueeze(0).to(args.device)

                    p = model(image).softmax(-1)
                    pred, p = model.tokenizer.decode(p)

                    # Store ground truth and predicted texts
                    groundtruth_texts.append(groundtruth_text)
                    predicted_texts.append(pred[0])

                # Calculate CRR and WRR
                crr, wrr = calculate_crr_wrr(groundtruth_texts, predicted_texts)

                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"Dataset: {dataset_dir}\n")
                    out_f.write(f"\tCharacter Recognition Rate (CRR): {crr:.4f}\n")
                    out_f.write(f"\tWord Recognition Rate (WRR): {wrr:.4f}\n")
                print(f"Processed dataset: {dataset_dir} - CRR/WRR: {(crr)*100:.2f} / {(wrr)*100:.2f}")
            else:
                print(f"No processed_for_eval.txt file found in {dataset_dir}") 

    
if __name__ == '__main__':
    main()