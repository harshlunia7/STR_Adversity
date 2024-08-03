import numpy as np
import random
import cv2
import math
import os
import glob
import json
import matplotlib.pyplot as plt
from ..base_degradation import Degradation

class FogDegradation(Degradation):
    def __init__(self, degree, image_type):
        super().__init__(degree, image_type)
        self.atmospheric_light = 0.5 * 255
        self.beta_range = 0.01 * np.array([130, 230], dtype=np.float64) + 0.05 # array([1.35, 2.35])

    def get_scene_depth(self, image_shape):
        row_indices, col_indices = np.indices(image_shape) + 1
        euclidean_distance_from_center = np.sqrt((row_indices - image_shape[0]//2)**2 + (col_indices - image_shape[1]//2)**2)
        # scene_depth based on below formula will produce a depth map with points in the middle being further away(higher value) 
        scene_depth = (-0.04) * euclidean_distance_from_center + np.sqrt(np.maximum(image_shape[0], image_shape[1]))
        # Need Clarity for below inversion opperation Using this nature of scene depth map was resulting in a behaviour opposite to expectation from atmospheric scattering formula -> furthest points(middle pixels) was less foggy (opposite of expectation)
        normalized_depth_map = (255 - np.clip(scene_depth, 0, 255)) / 255
        return normalized_depth_map
    
    """Concrete class for Fog degradation."""
    def apply(self, image):
        image = np.array(image, dtype=np.float64)
        image_shape = image.shape[0:2] # w,h
        scene_depth = self.get_scene_depth(image_shape)
        transmission_map = (np.exp(-self.beta_range[self.degree]) * scene_depth)[:,:, np.newaxis]
        foggy_image = np.multiply(image, transmission_map) + self.atmospheric_light * (1 - transmission_map)
        return np.clip(foggy_image, 0 , 255)
    