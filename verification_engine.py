import cv2
import hashlib

from embed_dwt_dct_svd import EmbedDwtDctSvd
from alignment import ImageAligner
from advanced_metrics import AdvancedMetrics


class VerificationEngine:

    def __init__(self, db):
        self.db = db

    # ---------------------------------------
    # VERIFY TAMPERED IMAGE
    # ---------------------------------------
    def verify(self, tampered_image_path):

        attacked = cv2.imread(tampered_image_path)

        if attacked is None:
            raise Exception("Invalid uploaded image")

        attacked_hash = hashlib.sha256(attacked.tobytes()).hexdigest()

        candidates = self.db.images  # using DummyDB structure

        for image_id, record in candidates.items():

            original = cv2.imread(record["watermarked_path"])

            # 1️⃣ Geometric Alignment
            aligned = ImageAligner.align(original, attacked)

            # 2️⃣ Rebuild watermark bits
            user = self.db.get_user(record["user_id"])

            from watermark_builder import WatermarkBuilder

            builder = WatermarkBuilder()
            watermark_bits = builder.build_dynamic_watermark(
                record["user_id"],
                image_id,
                user["signature_bits"],
                record["timestamp"],
                record["image_hash"]
            )

            # 3️⃣ Decode
            embedder = EmbedDwtDctSvd(watermark_bits)
            extracted_bits = embedder.decode(aligned)

            # 4️⃣ Similarity (NC)
            nc = AdvancedMetrics.nc(watermark_bits, extracted_bits)
            ber = AdvancedMetrics.ber(watermark_bits, extracted_bits)

            # 5️⃣ Decide tampered or not
            if nc > 0.90:

                if attacked_hash != record["image_hash"]:
                    status = "COPY_ATTACK"
                else:
                    status = "AUTHENTIC"

            else:
                status = "FORGED_OR_TAMPERED"

            # 6️⃣ Robustness Metrics
            features = {
                "PSNR": AdvancedMetrics.psnr(original, attacked),
                "SSIM": AdvancedMetrics.ssim_score(original, attacked),
                "NC": nc,
                "BER": ber,
                "Entropy_Diff": AdvancedMetrics.entropy_difference(original, attacked),
                "Mean_Diff": AdvancedMetrics.mean_intensity_difference(original, attacked),
                "Variance_Diff": AdvancedMetrics.variance_difference(original, attacked),
                "Histogram_Corr": AdvancedMetrics.histogram_correlation(original, attacked),
                "HF_Energy_Ratio": AdvancedMetrics.high_frequency_energy_ratio(attacked),
                "DCT_Variance": AdvancedMetrics.dct_variance(attacked),
                "Wavelet_Energy_Change": AdvancedMetrics.wavelet_energy_change(original, attacked),
                "Laplacian_Var": AdvancedMetrics.laplacian_variance(attacked),
                "Attack_Type": self.detect_attack_type(features=None),  # optional ML later
                "Status": status
            }

            # 7️⃣ Store robustness results
            self.db.store_robustness(image_id, features)

            return {
                "status": status,
                "image_id": image_id,
                "metrics": features
            }

        return {"status": "UNKNOWN_IMAGE"}

    # ---------------------------------------
    # Basic Heuristic Attack Type Detection
    # ---------------------------------------
    def detect_attack_type(self, features):

        # Optional simple rules (can replace with ML model later)
        return "UNKNOWN"