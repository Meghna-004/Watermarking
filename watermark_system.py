import cv2
import hashlib
from datetime import datetime

from embed_dwt_dct_svd import EmbedDwtDctSvd
from watermark_builder import WatermarkBuilder


class WatermarkSystem:

    def __init__(self, db):
        self.db = db  # database handler

    # -------------------------------------
    # EMBED WATERMARK
    # -------------------------------------

    def embed(self, user_id, input_path, output_path):

        # 1️⃣ Fetch user data from DB
        user = self.db.get_user(user_id)

        if user is None:
            raise Exception("User not found")

        signature_bits = user["signature_bits"]
        unique_id = user_id

        # 2️⃣ Prepare metadata
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_id = self.db.generate_image_id()

        image = cv2.imread(input_path)
        if image is None:
            raise Exception("Invalid input image")

        image_hash = hashlib.sha256(image.tobytes()).hexdigest()

        # 3️⃣ Build secure dynamic watermark
        builder = WatermarkBuilder()
        watermark_bits = builder.build_dynamic_watermark(
            unique_id,
            image_id,
            signature_bits,
            timestamp,
            image_hash
        )

        # 4️⃣ Embed watermark using DWT-DCT-SVD
        embedder = EmbedDwtDctSvd(watermark_bits)
        watermarked = embedder.encode(image)

        cv2.imwrite(output_path, watermarked)

        # 5️⃣ Store metadata in DB
        self.db.store_image({
            "image_id": image_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "image_hash": image_hash,
            "original_path": input_path,
            "watermarked_path": output_path
        })

        return {
            "status": "EMBED_SUCCESS",
            "image_id": image_id,
            "timestamp": timestamp
        }