# import cv2
# from id_generator import UniqueIDGenerator
# from watermark_builder import WatermarkBuilder
# from embed_dwt_dct_svd import EmbedDwtDctSvd


# class WatermarkSystem:

#     def __init__(self):
#         self.unique_id = None
#         self.watermark_bits = None

#     def generate_user_watermark(self, signature_path):
#         self.unique_id = UniqueIDGenerator.generate()

#         builder = WatermarkBuilder()
#         self.watermark_bits = builder.build_from_id_and_signature(
#             self.unique_id, signature_path
#         )

#         return self.unique_id

#     def embed(self, input_path, output_path):
#         image = cv2.imread(input_path)
#         embedder = EmbedDwtDctSvd(self.watermark_bits)
#         watermarked = embedder.encode(image)
#         cv2.imwrite(output_path, watermarked)

#     def decode(self, image_path):
#         image = cv2.imread(image_path)
#         embedder = EmbedDwtDctSvd(self.watermark_bits)
#         return embedder.decode(image)


import cv2
from id_generator import UniqueIDGenerator
from watermark_builder import WatermarkBuilder
from embed_dwt_dct_svd import EmbedDwtDctSvd


class WatermarkSystem:

    def __init__(self):
        self.unique_id = None
        self.watermark_bits = None

    def generate_user_watermark(self, signature_path, image_name):
        self.unique_id = UniqueIDGenerator.generate()

        builder = WatermarkBuilder()
        self.watermark_bits = builder.build_dynamic_watermark(
            self.unique_id,
            image_name,
            signature_path
        )

        return self.unique_id

    def embed(self, input_path, output_path):
        image = cv2.imread(input_path)
        embedder = EmbedDwtDctSvd(self.watermark_bits)
        watermarked = embedder.encode(image)
        cv2.imwrite(output_path, watermarked)

    def decode(self, image_path):
        image = cv2.imread(image_path)
        embedder = EmbedDwtDctSvd(self.watermark_bits)
        return embedder.decode(image)

    # ✅ ADD THIS
    def verify(self, image_path, threshold=0.95):

        extracted_bits = self.decode(image_path)

        if self.watermark_bits is None:
            raise Exception("Watermark not generated!")

        min_len = min(len(self.watermark_bits), len(extracted_bits))

        matches = 0
        for i in range(min_len):
            if self.watermark_bits[i] == extracted_bits[i]:
                matches += 1

        similarity = matches / min_len

        print("Similarity Score:", similarity)

        if similarity >= threshold:
            return "AUTHENTIC (Not Tampered)"
        else:
            return "TAMPERED"