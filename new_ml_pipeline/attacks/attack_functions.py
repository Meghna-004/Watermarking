# attack_functions.py

import cv2
import numpy as np

STRENGTH_LEVELS = {

    "GAUSSIAN_NOISE":[0.01,0.03,0.05],

    "SALT_PEPPER":[0.01,0.03,0.05],

    "BLUR":[3,5,7],

    "ROTATION":[5,10,20],

    "CROP":[0.05,0.1,0.2],

    "SCALING":[0.9,0.7,0.5]
}

def gaussian_noise(img,var):

    noise = np.random.normal(0, np.sqrt(var)*255, img.shape)

    noisy = img + noise

    return np.clip(noisy,0,255).astype(np.uint8)


def salt_pepper(img,amount):

    out=img.copy()

    h,w=img.shape[:2]

    num=int(amount*h*w/2)

    ys=np.random.randint(0,h,num)
    xs=np.random.randint(0,w,num)

    out[ys,xs]=255

    ys=np.random.randint(0,h,num)
    xs=np.random.randint(0,w,num)

    out[ys,xs]=0

    return out


def blur_attack(img,k):

    return cv2.GaussianBlur(img,(k,k),0)


def rotation_attack(img,angle):

    h,w=img.shape[:2]

    M=cv2.getRotationMatrix2D((w//2,h//2),angle,1)

    return cv2.warpAffine(
        img,
        M,
        (w,h),
        borderMode=cv2.BORDER_REFLECT
    )


def crop_attack(img,ratio):

    h,w=img.shape[:2]

    nh = int(h * (1 - ratio))
    nw = int(w * (1 - ratio))
 
    if nh <= 0 or nw <= 0 or h == nh or w == nw:
        return img

    y = np.random.randint(0, h - nh)
    x = np.random.randint(0, w - nw)

    crop=img[y:y+nh,x:x+nw]

    if crop.size == 0:
        return img

    return cv2.resize(crop,(w,h))


def scaling_attack(img,scale):

    h,w=img.shape[:2]

    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))

    small=cv2.resize(img,(new_w,new_h))

    return cv2.resize(small,(w,h))


def apply_attack(img,attack):

    level=np.random.choice(STRENGTH_LEVELS[attack])

    if attack=="GAUSSIAN_NOISE":
        img=gaussian_noise(img,level)

    elif attack=="SALT_PEPPER":
        img=salt_pepper(img,level)

    elif attack=="BLUR":
        img=blur_attack(img,level)

    elif attack=="ROTATION":
        img=rotation_attack(img,level)

    elif attack=="CROP":
        img=crop_attack(img,level)

    elif attack=="SCALING":
        img=scaling_attack(img,level)

    return img,level