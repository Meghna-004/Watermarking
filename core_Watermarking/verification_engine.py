# verification_engine.py
import cv2
import hashlib
import os
import numpy as np

from .embed_dwt_dct_svd import EmbedDwtDctSvd
from .alignment import ImageAligner
from .watermark_builder import WatermarkBuilder

from new_ml_pipeline.predict_attack import AttackPredictor
from new_ml_pipeline.features.forensic_features import compute_features


class VerificationEngine:

    def __init__(self, db):
        self.db = db
        self.predictor = AttackPredictor()

    # -----------------------------------
    # 🔹 INTERNAL METRICS (REPLACES advanced_metrics.py)
    # -----------------------------------
    @staticmethod
    def _nc(wm1, wm2):
        wm1 = np.array(wm1)
        wm2 = np.array(wm2)

        numerator = np.sum(wm1 * wm2)
        denominator = np.sqrt(np.sum(wm1 ** 2) * np.sum(wm2 ** 2))

        if denominator == 0:
            return 0

        return numerator / denominator

    @staticmethod
    def _ber(wm1, wm2):
        wm1 = np.array(wm1)
        wm2 = np.array(wm2)

        errors = np.sum(wm1 != wm2)
        return errors / len(wm1)

    # -----------------------------------
    # 🔹 MAIN VERIFY
    # -----------------------------------
    def verify(self, tampered_image_path, reference_image_id, user_id=None):

        attacked = cv2.imread(tampered_image_path)
        if attacked is None:
            raise Exception("Invalid uploaded image")

        record = self.db.images.get(reference_image_id)

        if not record:
            raise Exception("Reference image not found")

        # user validation (optional)
        if user_id and record["user_id"] != user_id:
            raise Exception("Unauthorized access to image")

        path = record.get("watermarked_path")

        if not path or not os.path.exists(path):
            raise Exception("Stored image not found")

        original = cv2.imread(path)
        if original is None:
            raise Exception("Invalid stored image")

        user = self.db.get_user(record["user_id"])
        if user is None:
            raise Exception("User not found")

        # ALIGN
        aligned = ImageAligner.align(original, attacked)

        # REBUILD WATERMARK
        builder = WatermarkBuilder()

        watermark_bits = builder.build_dynamic_watermark(
            record["user_id"],
            reference_image_id,
            user["signature_bits"],
            record["timestamp"],
            record["image_hash"]
        )

        # DECODE
        embedder = EmbedDwtDctSvd(watermark_bits)
        extracted_bits = embedder.decode(aligned)

        # METRICS
        nc = self._nc(watermark_bits, extracted_bits)
        ber = self._ber(watermark_bits, extracted_bits)

        attacked_hash = hashlib.sha256(attacked.tobytes()).hexdigest()

        if nc > 0.90:
            status = "AUTHENTIC" if attacked_hash == record["image_hash"] else "COPY_ATTACK"
        else:
            status = "FORGED_OR_TAMPERED"

        result = {
            "status": status,
            "image_id": reference_image_id,
            "NC": nc,
            "BER": ber
        }

        # ML + FEATURES (same as before)
        try:
            features = compute_features(original, attacked)

            features.update({
                "NC": nc,
                "BER": ber,
                "Status": status
            })

            result["metrics"] = features

            if status != "AUTHENTIC":
                result["Predicted_Attack"] = self.predictor.predict(original, attacked)
            else:
                result["Predicted_Attack"] = "NONE"

            result["summary"] = {
                "message": "Image is authentic" if status == "AUTHENTIC"
                        else "Image copied but watermark detected" if status == "COPY_ATTACK"
                        else "Image tampered or forged",
                "confidence": round(nc * 100, 2)
            }

        except Exception as e:
            result["Predicted_Attack"] = f"ERROR: {str(e)}"

        return result
