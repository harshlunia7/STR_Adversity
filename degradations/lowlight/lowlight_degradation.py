import numpy as np
import cv2
from ..base_degradation import Degradation

class LowLightDegradation(Degradation):
    def __init__(self, degree, image_type):
        super().__init__(degree, image_type)
        self.gamma_values = [2, 2.8]
        self.alpha_values = [0.65, 0.95]

    def calculate_average_intensity(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate the average intensity
        return np.mean(gray)

    def adaptive_gamma(self, average_intensity):
        # Map average intensity to a gamma value
        # This mapping can be adjusted as needed
        if self.degree == 0:
            return 2
        else:
            if average_intensity < 80:
                return np.random.uniform(2.4, 2.6)
            if average_intensity < 150:
                return np.random.uniform(2.7, 3.0)
            else:
                return np.random.uniform(3.2, 3.8)
    
    def adjust_gamma(self, image, gamma):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    
    """Concrete class for Fog degradation."""
    def apply(self, image):
        image = np.array(image, dtype=np.uint8)
        if self.image_type == "full":
            average_intensity = self.calculate_average_intensity(image)
            gamma = self.adaptive_gamma(average_intensity)
            print(gamma)
            dark_image = self.adjust_gamma(image, gamma)
        elif self.image_type == "word":
            image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            shadow_overlay = np.zeros_like(image_bgra, dtype=np.uint8)
            print("word", self.alpha_values[self.degree])

            dark_image = cv2.addWeighted(image_bgra, 1 - self.alpha_values[self.degree], shadow_overlay, self.alpha_values[self.degree], 0)
        return dark_image

    