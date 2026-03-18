#main.py
from ..watermark_system import WatermarkSystem
from ..signature_processor import SignatureProcessor
from ..id_generator import UniqueIDGenerator

# ----------------------------
# Dummy In-Memory DB
# ----------------------------

class DummyDB:

    def __init__(self):
        self.users = {}          # key = hashed_uid
        self.images = {}         # key = image_id
        self.image_counter = 1

    # ----------------------------
    # CREATE USER (Signup)
    # ----------------------------
    def create_user(self, signature_path):

        uid = UniqueIDGenerator.generate()
        hashed_uid = UniqueIDGenerator.secure_hash(uid)

        processor = SignatureProcessor()
        signature_bits = processor.image_to_bits(signature_path)

        self.users[hashed_uid] = {
            "uid_hash": hashed_uid,
            "signature_bits": signature_bits
        }

        print("\n[DB] User Stored:")
        print("Hashed UID:", hashed_uid[:20] + "...")
        print("Signature Bits Length:", len(signature_bits))

        return uid  # return original uid to user

    # ----------------------------
    # FETCH USER
    # ----------------------------
    def get_user(self, user_id):
        hashed_uid = UniqueIDGenerator.secure_hash(user_id)
        return self.users.get(hashed_uid)

    # ----------------------------
    # IMAGE ID GENERATOR
    # ----------------------------
    def generate_image_id(self):
        image_id = f"IMG_{self.image_counter}"
        self.image_counter += 1
        return image_id

    # ----------------------------
    # STORE IMAGE METADATA
    # ----------------------------
    def store_image(self, data):
        self.images[data["image_id"]] = data

        print("\n[DB] Image Stored:")
        print("Image ID:", data["image_id"])
        print("User ID (plain):", data["user_id"])
        print("Timestamp:", data["timestamp"])
        print("Image Hash (first 20 chars):", data["image_hash"][:20] + "...")

    # ----------------------------
    # STORE ROBUSTNESS METRICS
    # ----------------------------
    def store_robustness(self, image_id, features):

        if image_id not in self.images:
            print("[DB] Image not found for robustness logging")
            return

        if "robustness" not in self.images[image_id]:
            self.images[image_id]["robustness"] = []

        self.images[image_id]["robustness"].append(features)

        print("\n[DB] Robustness Stored for:", image_id)

    # ----------------------------
    # DEBUG PRINT FULL DB
    # ----------------------------
    def debug_print(self):

        print("\n================ DATABASE DUMP ================")

        print("\n--- USERS ---")
        for uid, data in self.users.items():
            print("Hashed UID:", uid[:20] + "...")
            print("Signature Bits Length:", len(data["signature_bits"]))
            print()

        print("\n--- IMAGES ---")
        for img_id, data in self.images.items():
            print("Image ID:", img_id)
            print("User ID:", data["user_id"])
            print("Timestamp:", data["timestamp"])
            print("Image Hash:", data["image_hash"][:20] + "...")
            print()

        print("===============================================")


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":

    db = DummyDB()

    # 1️⃣ Signup
    user_id = db.create_user("./assets/img_1.png")
    print("\nUser created with UID (given to user):", user_id)

    # 2️⃣ Watermark system
    system = WatermarkSystem(db)

    # 3️⃣ Embed watermark
    result = system.embed(
        user_id,
        "./assets/AI-generated.png",
        "./assets/AI-generatedW.jpg"
    )

    print("\nEmbed Result:", result)

    # 4️⃣ Show DB state
    db.debug_print()