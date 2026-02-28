# from id_generator import UniqueIDGenerator
# from signature_processor import SignatureProcessor
# from datetime import datetime


# class WatermarkBuilder:

#     def __init__(self, signature_size=32):
#         self.id_bits = []
#         self.signature_bits = []
#         self.signature_processor = SignatureProcessor(signature_size)

#     # # -------------------------------------------
#     # # 1️⃣ USER LEVEL WATERMARK (Same for all images)
#     # # -------------------------------------------
#     # def build_user_watermark(self, unique_id, signature_path):
#     #     hashed = UniqueIDGenerator.hash_id(unique_id)
#     #     self.id_bits = UniqueIDGenerator.string_to_bits(hashed)
#     #     self.signature_bits = self.signature_processor.image_to_bits(signature_path)
#     #
#     #     return self.id_bits + self.signature_bits
#     #
#     # # -------------------------------------------
#     # # 2️⃣ IMAGE LEVEL WATERMARK (Different per image)
#     # # -------------------------------------------
#     # def build_image_watermark(self, unique_id, image_name, signature_path):
#     #     dynamic_string = unique_id + image_name
#     #     hashed = UniqueIDGenerator.hash_id(dynamic_string)
#     #
#     #     self.id_bits = UniqueIDGenerator.string_to_bits(hashed)
#     #     self.signature_bits = self.signature_processor.image_to_bits(signature_path)
#     #
#     #     return self.id_bits + self.signature_bits

#     # -------------------------------------------
#     # 3️⃣ DYNAMIC WATERMARK (Different per upload)
#     # -------------------------------------------
#     def build_dynamic_watermark(self, unique_id, image_name, signature_path):

#         # Get formatted current timestamp
#         current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#         # Create dynamic string
#         dynamic_string = unique_id + image_name + current_timestamp

#         # Hash it
#         hashed = UniqueIDGenerator.hash_id(dynamic_string)

#         # Convert to bits
#         self.id_bits = UniqueIDGenerator.string_to_bits(hashed)
#         self.signature_bits = self.signature_processor.image_to_bits("./assets/signature.png")

#         return self.id_bits + self.signature_bits



from id_generator import UniqueIDGenerator
from signature_processor import SignatureProcessor
from datetime import datetime


class WatermarkBuilder:

    def __init__(self, signature_size=32):
        self.signature_processor = SignatureProcessor(signature_size)

    def build_dynamic_watermark(self, unique_id, image_name, signature_path):

        # Current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        dynamic_string = unique_id + image_name + current_timestamp
        hashed = UniqueIDGenerator.hash_id(dynamic_string)

        id_bits = UniqueIDGenerator.string_to_bits(hashed)
        signature_bits = self.signature_processor.image_to_bits(signature_path)

        return id_bits + signature_bits