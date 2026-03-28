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
    def verify(self, tampered_image_path, user_id=None):

        attacked = cv2.imread(tampered_image_path)
        if attacked is None:
            raise Exception("Invalid uploaded image")

        attacked_hash = hashlib.sha256(attacked.tobytes()).hexdigest()

        best_result = None
        best_nc = -1

        # Fetch user once
        user_data = self.db.get_user(user_id) if user_id else None

        # Initialize builder once
        builder = WatermarkBuilder()

        # --------------------------------------------------
        # 🔍 FAST MATCHING LOOP
        # --------------------------------------------------
        for image_id, record in self.db.images.items():

            # USER FILTER
            if user_id is not None and record["user_id"] != user_id:
                continue

            path = record.get("watermarked_path")

            if not path or not os.path.exists(path):
                continue

            original = cv2.imread(path)
            if original is None:
                continue

            # Safe user fetch
            user = user_data if user_id else self.db.get_user(record["user_id"])
            if user is None:
                continue

            # Always align (more robust)
            aligned = ImageAligner.align(original, attacked)

            # 🔹 Rebuild watermark
            watermark_bits = builder.build_dynamic_watermark(
                record["user_id"],
                image_id,
                user["signature_bits"],
                record["timestamp"],
                record["image_hash"]
            )

            # 🔹 Decode
            embedder = EmbedDwtDctSvd(watermark_bits)
            extracted_bits = embedder.decode(aligned)

            # 🔹 Compute metrics (inline)
            nc = self._nc(watermark_bits, extracted_bits)
            ber = self._ber(watermark_bits, extracted_bits)

            # 🔹 Status
            if nc > 0.90:
                status = "AUTHENTIC" if attacked_hash == record["image_hash"] else "COPY_ATTACK"
            else:
                status = "FORGED_OR_TAMPERED"

            # Normal best selection
            if nc > best_nc:
                best_nc = nc
                best_result = {
                    "status": status,
                    "image_id": image_id,
                    "NC": nc,
                    "BER": ber
                }

            # Early exit (optimization)
            if nc > 0.95:
                break

        # --------------------------------------------------
        # 🎯 FINAL PROCESSING (HEAVY WORK ONLY ONCE)
        # --------------------------------------------------
        if best_result and best_nc > 0.6:

            try:
                best_path = self.db.images[best_result["image_id"]]["watermarked_path"]
                original = cv2.imread(best_path)

                # Compute features once
                features = compute_features(original, attacked)

                features.update({
                    "NC": best_result["NC"],
                    "BER": best_result["BER"],
                    "Status": best_result["status"]
                })

                best_result["metrics"] = features

                # ML Prediction once
                if best_result["status"] != "AUTHENTIC":
                    best_result["Predicted_Attack"] = self.predictor.predict(original, attacked)
                else:
                    best_result["Predicted_Attack"] = "NONE"

                # FRONTEND-FRIENDLY SUMMARY
                confidence = round(best_result["NC"] * 100, 2)

                if best_result["status"] == "AUTHENTIC":
                    message = "Image is authentic"
                elif best_result["status"] == "COPY_ATTACK":
                    message = "Image copied but watermark detected"
                else:
                    message = "Image tampered or forged"

                best_result["summary"] = {
                    "message": message,
                    "confidence": confidence
                }

            except Exception as e:
                best_result["Predicted_Attack"] = f"ERROR: {str(e)}"

            return best_result

        return {"status": "UNKNOWN_IMAGE"}
