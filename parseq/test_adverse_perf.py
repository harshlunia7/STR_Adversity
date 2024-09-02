import argparse
import  glob
import os

import torch
from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
# from strhub.data.utils import CharsetAdapter
from nltk import edit_distance


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--word_image_path', help='word images path')
    parser.add_argument('--gt_path', help='word image gt file path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image_ext', default='jpeg')
    parser.add_argument('--charset', default='')
    parser.add_argument('--label', default='')


    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    # charset_adapter = CharsetAdapter( " ೂ ು ೃ ಃ ೌ ೋ ೕ ೇ ೊ ೈ ೖ ೀ ೄ ೆ ್ ಾ ಼ ಿ ಂ ಁೲ ೢ ೣಀಪ೧೩೪ಧಒಆಞಣಽಈಗ೭ಯಉಛಖಏಚಓಫ೫ರಬ೦ಠಔಊಐಮಌಘ೬ತಭಜಋದಶಎನಕಇ೮೯ಅ೨ಸೠಷಟವಱಳಲಥೡಹಙೞಢಡಝೱ\u0cf3 ")
    # print(f"CHECK: {charset_adapter('শলাং')}")

    gt_dict = {}
    with open(args.gt_path) as gt_obj:
        for entry in gt_obj:
            gt_dict[entry.strip().split("\t")[0]] = entry.strip().split("\t")[-1]


    # model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    # model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True, skip_validation=True).eval().to(args.device)
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    correct = 0
    total = 0   
    ned = 0
    label_length = 0
    for image_file_path in glob.glob(os.path.join(args.word_image_path, f"*.{args.image_ext}")):
        # Load image and prepare for input
        image_file_name = image_file_path.split('/')[-1]

        image = Image.open(image_file_path).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        gt_text = gt_dict[image_file_name]

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)

        if pred[0] == gt_text:
            correct += 1
        total += 1
        ned += edit_distance(pred[0], gt_text) / max(len(pred[0]), len(gt_text))
        label_length += len(pred[0])

        # print(f'{image_file_name}: {pred[0]}')
    accuracy = correct / total
    total_ned = 1 - ned / total
    print(f"Accuracy: {accuracy} \nNED: {total_ned}\nTotal: {total}")

if __name__ == '__main__':
    main()

