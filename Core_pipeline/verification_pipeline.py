import os
import uuid
import cv2

from core_Watermarking.verification_engine import VerificationEngine
from new_ml_pipeline.attacks.apply_custom_attack import apply_multiple_attacks


class VerificationPipeline:

    def __init__(self, db):
        self.db = db
        self.engine = VerificationEngine(db)

    # -----------------------------------
    # 🟢 CASE (i): Upload attacked image
    # -----------------------------------
    def handle_uploaded_attacked_image(self, file, user_id=None):

        try:
            if not file:
                raise Exception("No file uploaded")

            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}.png"
            path = os.path.join(upload_dir, filename)

            file.save(path)

            # 🔹 Direct verification
            result = self.engine.verify(path, user_id)

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
    # 🔵 CASE (ii): Attack simulator
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

            # 🔹 Apply attacks
            attacked = apply_multiple_attacks(original, attack_list)

            # 🔹 Save attacked image
            attack_dir = "temp_attacks"
            os.makedirs(attack_dir, exist_ok=True)

            attack_path = os.path.join(attack_dir, f"{uuid.uuid4().hex}.png")
            cv2.imwrite(attack_path, attacked)

            # 🔹 Verify
            result = self.engine.verify(attack_path, user_id)

            return {
                "success": True,
                "type": "SIMULATION",
                "data": {
                    "attacked_path": attack_path,
                    "applied_attacks": attack_list,
                    "verification": result
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }