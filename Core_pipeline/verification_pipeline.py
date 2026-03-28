# verification_pipeline.py
import os
import uuid
import cv2

from core_Watermarking.verification_engine import VerificationEngine
from new_ml_pipeline.attacks.apply_custom_attack import apply_multiple_attacks
from new_ml_pipeline.attacks.apply_custom_attack import ATTACK_MENU


class VerificationPipeline:

    def __init__(self, db):
        self.db = db
        self.engine = VerificationEngine(db)

    # -----------------------------------
    # Utility: Convert frontend → backend format
    # -----------------------------------


    def convert_attack_list(self, frontend_attacks):
        """
        Converts:
        [
        { "type": "BLUR", "strength": 5 }
        ]
        →
        [
        ("BLUR", 5)
        ]
        """

        if not isinstance(frontend_attacks, list):
            raise Exception("Attacks must be a list")

        backend_attacks = []

        for attack in frontend_attacks:

            if not isinstance(attack, dict):
                raise Exception("Each attack must be an object")

            if "type" not in attack or "strength" not in attack:
                raise Exception("Invalid attack format. Required: type, strength")

            attack_type = attack["type"].upper()

            try:
                strength = float(attack["strength"])
            except ValueError:
                raise Exception("Strength must be a number")

            if attack_type not in ATTACK_MENU:
                raise Exception(f"Unsupported attack type: {attack_type}")

            low, high = ATTACK_MENU[attack_type]
            if not (low <= strength <= high):
                raise Exception(
                    f"{attack_type} strength must be between {low} and {high}"
                )

            if attack_type == "BLUR":
                strength = int(strength)

            backend_attacks.append((attack_type, strength))

        return backend_attacks

    # -----------------------------------
    # CASE (i): Upload attacked image
    # -----------------------------------
    def handle_uploaded_attacked_image(self, file, reference_image_id, user_id=None):

        try:
            if not file:
                raise Exception("No file uploaded")

            if not reference_image_id:
                raise Exception("Reference image is required")

            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}.png"
            path = os.path.join(upload_dir, filename)

            file.save(path)

            # verify with reference image
            result = self.engine.verify(path, reference_image_id, user_id)

            return {
                "success": True,
                "type": "UPLOAD_VERIFICATION",
                "data": {
                    "file_path": path,
                    "verification": result
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # -----------------------------------
    # CASE (ii): Attack simulator
    # -----------------------------------
    def handle_attack_simulation(self, image_id, attack_list, user_id=None):

        try:
            if not attack_list:
                raise Exception("No attacks provided")

            # 🔹 Get stored watermarked image
            record = self.db.images.get(image_id)

            if not record:
                raise Exception("Image not found")

            path = record["watermarked_path"]

            original = cv2.imread(path)
            if original is None:
                raise Exception("Invalid stored image")

            # 🔹 Convert frontend format → tuple format
            frontend_attacks = attack_list  # keep original for response
            backend_attacks = self.convert_attack_list(frontend_attacks)

            # 🔹 Apply attacks
            attacked = apply_multiple_attacks(original, backend_attacks)

            # 🔹 Save attacked image
            attack_dir = "temp_attacks"
            os.makedirs(attack_dir, exist_ok=True)

            attack_path = os.path.join(attack_dir, f"{uuid.uuid4().hex}.png")
            cv2.imwrite(attack_path, attacked)

            # 🔹 Verify
            result = self.engine.verify(attack_path, image_id, user_id)

            return {
                "success": True,
                "type": "SIMULATION",
                "data": {
                    "attacked_path": attack_path,
                    "applied_attacks": frontend_attacks,
                    "verification": result
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
