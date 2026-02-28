import uuid
import hashlib


class UniqueIDGenerator:

    @staticmethod
    def generate():
        return str(uuid.uuid4())

    @staticmethod
    def hash_id(unique_id):
        return hashlib.sha256(unique_id.encode()).hexdigest()

    @staticmethod
    def string_to_bits(data):
        binary = ''.join(format(ord(c), '08b') for c in data)
        return [int(b) for b in binary]