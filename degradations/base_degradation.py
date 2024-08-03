'''
Base Class: A base class that defines a common interface for all degradation types.
Concrete Classes: Concrete classes that inherit from the base class and implement specific types of degradation.
Factory Class: A factory class to instantiate the appropriate degradation class based on user input.
Main Program: The main program to handle user input, apply degradations, and save the results.

Degradation Base Class: Defines a common interface for all degradation types with an apply method that must be implemented by subclasses.
Concrete Classes: Implement specific degradations like GaussianBlurDegradation and SaltPepperNoiseDegradation.
Factory Class: Creates instances of the appropriate degradation class based on user input.
Main Program: Handles user input, applies the selected degradation to each image in the dataset, and saves the results to the specified output path.
'''


class Degradation:
    """Base class for all degradations."""
    def __init__(self, degree, image_type):
        self.degree = degree
        self.image_type = image_type
    
    def apply(self, image):
        raise NotImplementedError("Subclasses should implement this method")

