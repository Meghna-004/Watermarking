#signature_processor.py
import cv2
import numpy as np


class SignatureProcessor:

    def __init__(self, size=32):
        self.size = size

    def image_to_bits(self, path):
        """
        Used only during SIGNUP.
        Converts uploaded signature image to fixed bit array.
        """

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Invalid signature image")

        img = cv2.resize(img, (self.size, self.size))

        # Normalize lighting
        img = cv2.equalizeHist(img)

        # Binary threshold
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        return binary.flatten().astype(int).tolist()

    def array_to_bits(self, img_array):
        """
        Optional: if signature image comes as numpy array
        (e.g., from API upload instead of file path)
        """

        img = cv2.resize(img_array, (self.size, self.size))
        img = cv2.equalizeHist(img)

        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        return binary.flatten().astype(int).tolist()