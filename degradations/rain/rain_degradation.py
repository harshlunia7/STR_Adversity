import numpy as np
import random
import cv2
import math
import os
import json
import sys

from ..base_degradation import Degradation

class RainDegradation(Degradation):
    def __init__(self, degree, image_type):
        super().__init__(degree, image_type)
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        self.rain_mask_directories = [os.path.join(self.package_dir, '..', '..', 'data', 'rain_masks', 'medium_density'), 
                                      os.path.join(self.package_dir, '..', '..', 'data', 'rain_masks', 'high_density')]
        
    def add_rain_streaks(self, real_image, rain_mask):
        resized_rain_mask = cv2.resize(rain_mask, (real_image.shape[1], real_image.shape[0]))
        
        resized_rain_mask_3ch = cv2.merge([resized_rain_mask] * 3)
        normalized_rain_mask = resized_rain_mask_3ch / 255.0
        rainy_image = cv2.addWeighted(real_image, 1, resized_rain_mask_3ch, 0.5, 0)
        return rainy_image

    def apply(self, image):
        if self.image_type == "full":
            print("Not Implemented for Full Image!")
            sys.exit(0)
        
        rain_mask_filename = np.random.choice(os.listdir(self.rain_mask_directories[self.degree]))
        rain_mask_path = os.path.join(self.rain_mask_directories[self.degree], rain_mask_filename)
        rain_mask = cv2.imread(rain_mask_path, cv2.IMREAD_GRAYSCALE)

        rainy_image = self.add_rain_streaks(image, rain_mask)
        return rainy_image