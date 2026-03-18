# apply_custom_attack.py

import cv2

from .attack_functions import (
    gaussian_noise,
    salt_pepper,
    blur_attack,
    rotation_attack,
    crop_attack,
    scaling_attack
)


ATTACK_MENU = {
    "GAUSSIAN_NOISE": (0.01, 0.1),
    "SALT_PEPPER": (0.01, 0.1),
    "BLUR": (3, 11),
    "ROTATION": (1, 45),
    "CROP": (0.01, 0.3),
    "SCALING": (0.3, 1.0)
}


def apply_single_attack(img, attack, strength):

    if attack == "GAUSSIAN_NOISE":
        return gaussian_noise(img, strength)

    elif attack == "SALT_PEPPER":
        return salt_pepper(img, strength)

    elif attack == "BLUR":
        k = int(strength)
        if k % 2 == 0:
            k += 1
        return blur_attack(img, k)

    elif attack == "ROTATION":
        return rotation_attack(img, strength)

    elif attack == "CROP":
        return crop_attack(img, strength)

    elif attack == "SCALING":
        return scaling_attack(img, strength)

    else:
        raise ValueError("Invalid attack type")


def apply_multiple_attacks(img, attack_list):

    attacked = img.copy()

    for attack, strength in attack_list:
        attacked = apply_single_attack(attacked, attack, strength)

    return attacked


def get_user_attacks():

    print("\nAvailable Attacks:")
    for k, v in ATTACK_MENU.items():
        print(f"{k}: range {v}")

    n = int(input("\nNumber of attacks: "))

    attacks = []

    for i in range(n):
        attack = input(f"Attack {i+1}: ").strip().upper()

        while attack not in ATTACK_MENU:
            attack = input("Enter valid attack: ").strip().upper()

        low, high = ATTACK_MENU[attack]
        strength = float(input(f"Strength ({low}-{high}): "))

        attacks.append((attack, strength))

    return attacks