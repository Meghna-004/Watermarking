#id_generator
import uuid
import hashlib
import hmac

SECRET_KEY = b"MY_SUPER_SECRET_KEY"  # keep private


class UniqueIDGenerator:

    @staticmethod
    def generate():
        return str(uuid.uuid4())

    @staticmethod
    def secure_hash(data):
        return hmac.new(
            SECRET_KEY,
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    @staticmethod
    def string_to_bits(data):
        binary = ''.join(format(ord(c), '08b') for c in data)
        return [int(b) for b in binary]