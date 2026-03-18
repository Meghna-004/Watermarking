#watermark_builder.py
import hashlib
from .id_generator import UniqueIDGenerator


class WatermarkBuilder:

    def __init__(self):
        pass

    def build_dynamic_watermark(
        self,
        unique_id,
        image_id,
        signature_bits,
        timestamp,
        image_hash
    ):
        """
        Build watermark using:
        - Unique User ID
        - Image ID
        - Timestamp
        - Image content hash
        - Secret key (inside secure_hash)
        - Fixed signature bits from DB
        """

        # 🔐 Combine all security parameters
        dynamic_string = (
            unique_id +
            image_id +
            timestamp +
            image_hash
        )

        # 🔐 Secure hash (HMAC-based inside UniqueIDGenerator)
        hashed = UniqueIDGenerator.secure_hash(dynamic_string)

        # Convert to bit stream
        id_bits = UniqueIDGenerator.string_to_bits(hashed)

        # Final watermark = security bits + fixed signature bits
        watermark_bits = id_bits + signature_bits

        return watermark_bits