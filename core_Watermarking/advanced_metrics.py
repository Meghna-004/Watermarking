# advanced_metrics.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pywt


class AdvancedMetrics:

    # -----------------------------
    # PSNR
    # -----------------------------
    @staticmethod
    def psnr(img1, img2):
        return cv2.PSNR(img1, img2)


    # -----------------------------
    # SSIM
    # -----------------------------
    @staticmethod
    def ssim_score(img1, img2):

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(gray1, gray2, full=True)

        return score


    # -----------------------------
    # Normalized Correlation
    # -----------------------------
    @staticmethod
    def nc(wm1, wm2):

        wm1 = np.array(wm1)
        wm2 = np.array(wm2)

        numerator = np.sum(wm1 * wm2)

        denominator = np.sqrt(np.sum(wm1 ** 2) * np.sum(wm2 ** 2))

        if denominator == 0:
            return 0

        return numerator / denominator


    # -----------------------------
    # Bit Error Rate
    # -----------------------------
    @staticmethod
    def ber(wm1, wm2):

        wm1 = np.array(wm1)
        wm2 = np.array(wm2)

        errors = np.sum(wm1 != wm2)

        return errors / len(wm1)


    # -----------------------------
    # Entropy Difference
    # -----------------------------
    @staticmethod
    def entropy_difference(img1, img2):

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])

        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        entropy1 = -np.sum(hist1 * np.log2(hist1 + 1e-10))
        entropy2 = -np.sum(hist2 * np.log2(hist2 + 1e-10))

        return abs(entropy1 - entropy2)


    # -----------------------------
    # Mean Intensity Difference
    # -----------------------------
    @staticmethod
    def mean_intensity_difference(img1, img2):

        return abs(np.mean(img1) - np.mean(img2))


    # -----------------------------
    # Variance Difference
    # -----------------------------
    @staticmethod
    def variance_difference(img1, img2):

        return abs(np.var(img1) - np.var(img2))


    # -----------------------------
    # Histogram Correlation
    # -----------------------------
    @staticmethod
    def histogram_correlation(img1, img2):

        hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


    # -----------------------------
    # High Frequency Energy Ratio
    # -----------------------------
    @staticmethod
    def high_frequency_energy_ratio(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        magnitude = np.abs(fshift)

        h, w = magnitude.shape

        center = magnitude[h//4:3*h//4, w//4:3*w//4]

        high_freq = magnitude.sum() - center.sum()

        return high_freq / magnitude.sum()


    # -----------------------------
    # DCT Variance
    # -----------------------------
    @staticmethod
    def dct_variance(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dct = cv2.dct(np.float32(gray))

        return np.var(dct)


    # -----------------------------
    # Wavelet Energy Change
    # -----------------------------
    @staticmethod
    def wavelet_energy_change(img1, img2):

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        coeffs1 = pywt.dwt2(gray1, 'haar')
        coeffs2 = pywt.dwt2(gray2, 'haar')

        _, (LH1, HL1, HH1) = coeffs1
        _, (LH2, HL2, HH2) = coeffs2

        energy1 = np.sum(LH1**2 + HL1**2 + HH1**2)
        energy2 = np.sum(LH2**2 + HL2**2 + HH2**2)

        return abs(energy1 - energy2)


    # -----------------------------
    # Laplacian Variance (Blur Detection)
    # -----------------------------
    @staticmethod
    def laplacian_variance(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(gray, cv2.CV_64F).var()