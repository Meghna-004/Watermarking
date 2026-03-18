import os
import uuid

from core_Watermarking.watermark_system import WatermarkSystem


class WatermarkPipeline:

    def __init__(self, db):
        self.db = db
        self.system = WatermarkSystem(db)

    def process_upload(self, file, user_id):

        try:
            # ✅ Validate input
            if not file:
                raise Exception("No file uploaded")

            if not user_id:
                raise Exception("User ID is required")

            # 1. Save uploaded image
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}.png"
            input_path = os.path.join(upload_dir, filename)

            file.save(input_path)

            # 2. Output path
            output_dir = "storage/watermarked"
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"wm_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            # 3. Call watermark system
            result = self.system.embed(
                user_id,
                input_path,
                output_path
            )

            # 4. Add file path
            result["watermarked_path"] = output_path

            return {
                "success": True,
                "type": "WATERMARK_EMBED",
                "data": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }