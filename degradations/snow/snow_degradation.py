import numpy as np
import random
import cv2
import math
import os
import json
import sys


from ..base_degradation import Degradation

class SnowDegradation(Degradation):
    def __init__(self, degree, image_type):
        super().__init__(degree, image_type)
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        self.snow_mask_directories = [os.path.join(self.package_dir, '..', '..', 'data', 'snow_masks', 'mid_masks'), 
                                      os.path.join(self.package_dir, '..', '..', 'data', 'snow_masks', 'combined_masks')]
    
    def add_snow_streaks(self, real_image, snow_mask):
        resized_snow_mask = cv2.resize(snow_mask, (real_image.shape[1], real_image.shape[0]))
        
        resized_snow_mask_3ch = cv2.merge([resized_snow_mask] * 3)
        normalized_snow_mask = resized_snow_mask_3ch / 255.0
        snowy_image = cv2.addWeighted(real_image, 1, resized_snow_mask_3ch, 0.5, 0)
        return snowy_image
    
    def apply(self, image):
        if self.image_type == "full":
            print("Not Implemented for Full Image!")
            sys.exit(0)
        
        snow_mask_filename = np.random.choice(os.listdir(self.snow_mask_directories[self.degree]))
        snow_mask_path = os.path.join(self.snow_mask_directories[self.degree], snow_mask_filename)
        snow_mask = cv2.imread(snow_mask_path, cv2.IMREAD_GRAYSCALE)

        snowy_image = self.add_snow_streaks(image, snow_mask)
        return snowy_image