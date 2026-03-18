# forensic_features.py
import cv2
import numpy as np
import pywt

from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy


def jpeg_block_strength(gray):

    gray = gray.astype(np.float32)

    h,w = gray.shape

    block=8

    diff=0

    for i in range(block,h,block):

        diff+=np.sum(np.abs(gray[i,:]-gray[i-1,:]))

    for j in range(block,w,block):

        diff+=np.sum(np.abs(gray[:,j]-gray[:,j-1]))

    return diff/(h*w)


def scaling_artifact_score(gray):

    fft=np.fft.fft2(gray)

    fft_shift=np.fft.fftshift(fft)

    mag=np.abs(fft_shift)

    high_freq=mag>np.percentile(mag,90)

    vals = mag[high_freq]

    if vals.size == 0:
        return 0

    return np.mean(vals)


def compute_features(original,attacked):

    if original.shape != attacked.shape:
        attacked = cv2.resize(attacked,(original.shape[1],original.shape[0]))

    features={}

    gray_o=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    gray_a=cv2.cvtColor(attacked,cv2.COLOR_BGR2GRAY)

    features["PSNR"]=cv2.PSNR(original,attacked)

    features["SSIM"]=ssim(gray_o,gray_a,data_range=255)

    features["Entropy_Diff"]=abs(shannon_entropy(gray_a)-shannon_entropy(gray_o))

    features["Mean_Diff"]=np.mean(attacked)-np.mean(original)

    features["Variance_Diff"]=np.var(attacked)-np.var(original)

    # Detects Gaussian / Salt & Pepper noise
    features["Noise_Variance"]=np.var(gray_a - gray_o)

    # Median residual (detects noise artifacts)
    median = cv2.medianBlur(gray_a,3)
    features["Median_Residual"] = np.mean(np.abs(gray_a - median))

    hist1=cv2.calcHist([gray_o],[0],None,[256],[0,256])
    hist2=cv2.calcHist([gray_a],[0],None,[256],[0,256])

    features["Histogram_Corr"]=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

    lap=cv2.Laplacian(gray_a,cv2.CV_64F)

    features["Laplacian_Var"]=lap.var()

    edges=cv2.Canny(gray_a,100,200)

    features["Edge_Density"]=np.sum(edges>0)/edges.size

    gx=cv2.Sobel(gray_a,cv2.CV_64F,1,0)
    gy=cv2.Sobel(gray_a,cv2.CV_64F,0,1)

    features["Tenengrad_Score"]=np.mean(gx**2+gy**2)

    dct=cv2.dct(np.float32(gray_a)/255.0)

    hf=dct[gray_a.shape[0]//2:,gray_a.shape[1]//2:]

    den = np.sum(np.abs(dct))

    if den == 0:
        features["HF_Energy_Ratio"] = 0
    else:
        features["HF_Energy_Ratio"] = np.sum(np.abs(hf)) / den

    features["DCT_Variance"]=np.var(dct)

    coeffs=pywt.dwt2(gray_a,'haar')

    LL,(LH,HL,HH)=coeffs

    features["Wavelet_Energy_Change"]=np.sum(np.abs(LH))+np.sum(np.abs(HL))+np.sum(np.abs(HH))

    # features["Blockiness_Metric"]=jpeg_block_strength(gray_a)

    features["JPEG_Block_Strength"]=jpeg_block_strength(gray_a)

    features["Scaling_Artifact_Score"]=scaling_artifact_score(gray_a)

    return features