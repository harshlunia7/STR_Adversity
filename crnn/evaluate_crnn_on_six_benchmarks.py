import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import models.crnn as crnn
import params
import argparse

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
    parser.add_argument('-i', '--input', type = str, required = True, help = 'evaluation datasets directory')
    parser.add_argument('-g', '--gt_file', type = str, required=True, help = 'gt file')

    args = parser.parse_args()

    model_path = args.model_path

    # net init
    nclass = len(params.alphabet) + 1
    model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)

    model.apply(weights_init)
    model_dict = model.state_dict()

    if torch.cuda.is_available():
        model = model.cuda()
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]

    model_dict.update(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    # load model
    print('loading pretrained model from %s' % model_path)
    # if params.multi_gpu:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(params.alphabet)
    transformer = dataset.resizeNormalize((100, 32))

    output_file = os.path.join(args.input, f"evaluated_CRNN_crr_wrr.txt")

    for dataset_dir in os.listdir(args.input):
        dataset_path = os.path.join(args.input, dataset_dir)
        print(dataset_dir, dataset_path)

        if os.path.isdir(dataset_path):
            processed_text_file = os.path.join(dataset_path, os.path.basename(args.gt_file))
            print(processed_text_file)
            # Find the processed_for_eval.txt file in the current dataset directory
            # for file_name in os.listdir(dataset_path):
            #     if file_name.endswith('processed_for_eval.txt'):
            #         processed_text_file = os.path.join(dataset_path, file_name)
            #         break

            if processed_text_file:
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

                    image = Image.open(image_path).convert('L')
                    image = transformer(image)
                    if torch.cuda.is_available():
                        image = image.cuda()
                    image = image.view(1, *image.size())
                    image = Variable(image)

                    preds = model(image)
                    # print(preds)

                    _, preds = preds.max(2)
                    # preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds = preds.squeeze(1)

                    preds_size = Variable(torch.LongTensor([preds.size(0)]))
                    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

                    # Store ground truth and predicted texts
                    groundtruth_texts.append(groundtruth_text)
                    predicted_texts.append(sim_pred)
                    count_records_processed += 1


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
