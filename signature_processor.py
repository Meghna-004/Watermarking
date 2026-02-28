import cv2


class SignatureProcessor:

    def __init__(self, size=32):
        self.size = size

    def image_to_bits(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (self.size, self.size))
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return binary.flatten().tolist()