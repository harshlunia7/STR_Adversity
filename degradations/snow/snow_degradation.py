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
        self.number_of_streaks = [70, 90]
        self.line_width = [1, 2]
    
    def apply(self, image):
        if self.image_type == "full":
            print("Not Implemented for Full Image!")
            sys.exit(0)
        
        w, h, n_channels = image.shape
        isgray = n_channels == 1

        line_width = self.line_width[self.width]
        n_rains = self.rng.integers(self.number_of_streaks[self.degree], self.number_of_streaks[self.degree] + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.)
            y2 = y1 + length * math.cos(slant * math.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(image, (x1, y1), (x2, y2), fillcolor, line_width)

        return image