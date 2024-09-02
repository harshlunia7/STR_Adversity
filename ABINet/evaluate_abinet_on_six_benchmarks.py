import os
import argparse
import logging
import os
import torch
import PIL
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import Config, Logger, CharsetMapper

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

# Placeholder function to get the predicted text given an image path
def get_predicted_text(image_path):
    # Replace this function's content with the actual model inference logic
    # For now, it returns a dummy prediction (just as a placeholder)
    return "predicted_text"

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
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='figs/test')
    parser.add_argument('--gt_file', type=str, default='test.txt')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='workdir/train-abinet/best-train-abinet.pth')
    parser.add_argument('--model_eval', type=str, default='alignment', 
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)

    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f"cuda:{args.cuda}"

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)

    logging.info('Construct model.')
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    
    output_file = os.path.join(args.input, f"evaluated_abinet_crr_wrr_{os.path.basename(args.gt_file).split('.')[0]}.txt")
    

    for dataset_dir in os.listdir(args.input):
        dataset_path = os.path.join(args.input, dataset_dir)

        if os.path.isdir(dataset_path):
            processed_text_file = os.path.join(dataset_path, os.path.basename(args.gt_file))
            print(processed_text_file)
            # Find the processed_for_eval.txt file in the current dataset directory
            # for file_name in os.listdir(dataset_path):
            #     if file_name.endswith('processed_for_eval.txt'):
            #         processed_text_file = os.path.join(dataset_path, file_name)
            #         break

            if os.path.isfile(processed_text_file):
                groundtruth_texts = []
                predicted_texts = []

                with open(processed_text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                count_records = 0
                for line in lines:
                    if count_records % 100 == 0:
                        print(f"Records done: {count_records}")
                    image_filename, groundtruth_text = line.strip().split('\t')
                    image_path = os.path.join(dataset_path, image_filename)

                    img = PIL.Image.open(image_path).convert('RGB')
                    img = preprocess(img, config.dataset_image_width, config.dataset_image_height)
                    img = img.to(device)
                    res = model(img)
                    pt_text, _, __ = postprocess(res, charset, config.model_eval)
                    
                    # Store ground truth and predicted texts
                    groundtruth_texts.append(groundtruth_text)
                    predicted_texts.append(pt_text[0])
                    count_records += 1
                
                # Calculate CRR and WRR
                crr, wrr = calculate_crr_wrr(groundtruth_texts, predicted_texts)

                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"Dataset: {dataset_dir} GT File {os.path.basename(args.gt_file)}\n")
                    out_f.write(f"\tCharacter Recognition Rate (CRR): {crr:.4f}\n")
                    out_f.write(f"\tWord Recognition Rate (WRR): {wrr:.4f}\n")
                
                print(f"Processed dataset: {dataset_dir} - CRR/WRR: {(crr)*100:.2f} / {(wrr)*100:.2f}")
            else:
                print(f"No processed_for_eval.txt file found in {dataset_dir}")   

if __name__ == '__main__':
    main()

        
     
