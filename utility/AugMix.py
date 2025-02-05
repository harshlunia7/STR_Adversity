# ==============================================================================
# The source of the below code is AugMix 
#  Code : https://github.com/google-research/augmix
#  Paper: https://arxiv.org/pdf/1912.02781.pdf
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations
import numpy as np
import random
from PIL import Image # HARSH :: Python image processing library, QUESTION:: Why not OpenCV?? Guess all operations in augmentations from the above paper are implemented in PIL, would find the answer by reading the paper

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  '''
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)
  '''
  return image

def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image # HARSH:: Maybe to have it in RGB format by default as Opencv reads in BGR by default NEED TO CONFIRM
  pil_img = op(pil_img, severity) # HARSH:: apply the operation to the image
  return np.asarray(pil_img) / 255.


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain                                              # HARSH:: Number of augmentation iterations to be applied on the image as a whole
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly       # HARSH:: Under each iteration how may operations need to be applied can be 2 or 3 
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image)
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(2, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations.augmentations)
      #print(op)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * normalize(image_aug)
    
  max_ws = max(ws)
  rate = 1.0 / max_ws
  #print(rate)
  
  
  #mixed = (random.randint(5000, 9000)/10000) * normalize(image) + (random.randint((int)(rate*3000), (int)(rate*10000))/10000) * mix
  mixed = max((1 - m), 0.7) * normalize(image) + max(m, rate*0.5) * mix
  #mixed = (1 - m) * normalize(image) + m * mix
  return mixed

