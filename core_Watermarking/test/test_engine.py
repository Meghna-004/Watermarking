import cv2

from core_Watermarking.test.main import DummyDB
from core_Watermarking.watermark_system import WatermarkSystem
from core_Watermarking.verification_engine import VerificationEngine

from new_ml_pipeline.attacks.apply_custom_attack import (
    apply_multiple_attacks,
    get_user_attacks
)


INPUT_IMAGE = "assets/og_image.png"
WATERMARKED_IMAGE = "assets/watermarked_output.jpg"
ATTACKED_IMAGE = "assets/attacked_output.jpg"


def run_test():

    db = DummyDB()

    user_id = db.create_user("assets/img_1.png")

    wm_system = WatermarkSystem(db)

    wm_system.embed(
        user_id,
        INPUT_IMAGE,
        WATERMARKED_IMAGE
    )

    watermarked = cv2.imread(WATERMARKED_IMAGE)

    # User-defined attacks
    attack_list = get_user_attacks()

    attacked = apply_multiple_attacks(watermarked, attack_list)

    cv2.imwrite(ATTACKED_IMAGE, attacked)

    verifier = VerificationEngine(db)
    result = verifier.verify(ATTACKED_IMAGE)

    print("\nResult")
    print("Status:", result["status"])
    print("Image ID:", result["image_id"])
    print("Predicted Attack:", result["Predicted_Attack"])


if __name__ == "__main__":
    run_test()