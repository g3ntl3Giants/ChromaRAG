from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Function to encrypt sensitive data
def encrypt_data(data: str) -> str:
    encrypted_data = cipher_suite.encrypt(data.encode('utf-8'))
    return encrypted_data.decode('utf-8')

# Function to decrypt sensitive data
def decrypt_data(encrypted_data: str) -> str:
    decrypted_data = cipher_suite.decrypt(encrypted_data.encode('utf-8'))
    return decrypted_data.decode('utf-8')

# Function to anonymize sensitive data by hashing it
def anonymize_data(data: str) -> str:
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(data.encode('utf-8'))
    return digest.finalize().hex()
