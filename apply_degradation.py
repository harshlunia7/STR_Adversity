import os
import cv2
import numpy as np
from degradations.degradation_factory import DegradationFactory
from utility.miscellaneous import get_all_images, save_image_in_path

def process_dataset(args):
    """Main function to process the dataset with the specified degradation."""
    input_path, image_at_level, = args.input_path, args.image_at_level
    output_path = args.output_path
    degradation_type, degree, image_type = args.degradation_type, args.degree, args.image_type
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    factory = DegradationFactory()
    degradation = factory.get_degradation(degradation_type, degree, image_type)
    image_paths = get_all_images(input_path, image_at_level)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        degraded_image_save_location = os.path.relpath(image_path, input_path)
        degraded_image = degradation.apply(image)
        save_image_in_path(os.path.join(output_path, degraded_image_save_location), degraded_image)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply degradation to a dataset.")
    parser.add_argument("degradation_type", type=str, help="Type of degradation to apply.")
    parser.add_argument("degree", type=int, choices=[0, 1, 2, 3], help="Degree of degradation.")
    parser.add_argument("image_type", type=str, choices=["word", "full"], help="Word-Image or Full size Images.")
    parser.add_argument("--input_path", type=str, help="Path where all the non-degraded images are located")
    parser.add_argument("--image_at_level", type=int, default=-1, help="Number of levels to look for images in input path")
    parser.add_argument("--output_path", type=str, default="./output_samples", help="Path to save all the degraded images")

    args = parser.parse_args()

    process_dataset(args)
