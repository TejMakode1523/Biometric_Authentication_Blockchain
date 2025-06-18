import numpy as np
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Load the embeddings
fingerprint_embeddings = np.load("fingerprint_embeddings.npy", allow_pickle=True).item()
facial_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()

# Ensure both dictionaries have the same keys (identities)
assert (
    fingerprint_embeddings.keys() == facial_embeddings.keys()
), "Mismatch in identity keys"

# Combine the embeddings for each identity, ensuring they have the same dimensions
combined_embeddings = {
    key: np.concatenate(
        (
            np.ravel(fingerprint_embeddings[key]),  # Flatten the fingerprint embedding
            np.ravel(facial_embeddings[key]),  # Flatten the facial embedding
        )
    )
    for key in fingerprint_embeddings.keys()
}


# Save combined embeddings to a file
def save_combined_embeddings(embeddings, filename):
    np.save(filename, embeddings)


save_combined_embeddings(combined_embeddings, "combined_embeddings.npy")

# Use a fixed key for encryption (must be 32 bytes for AES-256)
fixed_key = b"ThisIsA32ByteFixedKeyForAES256!!"


def encrypt_data(key, data):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    ciphertext = cipher.encrypt(pad(data, AES.block_size))
    return iv + ciphertext


def decrypt_data(key, encrypted_data):
    iv = encrypted_data[: AES.block_size]
    ciphertext = encrypted_data[AES.block_size :]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return plaintext


# Encrypt the combined embeddings and save to separate text files
for key in combined_embeddings.keys():
    combined_embedding_bytes = combined_embeddings[key].astype(np.float32).tobytes()
    encrypted_data = encrypt_data(fixed_key, combined_embedding_bytes)

    # Convert the encrypted data to Base64 for text storage
    encrypted_data_base64 = base64.b64encode(encrypted_data).decode("utf-8")

    # Save to a text file named after the user key
    with open(f"{key}_encrypted.txt", "w") as file:
        file.write(encrypted_data_base64)

# Decrypt and verify
decrypted_embeddings = {}
for key in combined_embeddings.keys():
    # Read the encrypted data from the text file
    with open(f"{key}_encrypted.txt", "r") as file:
        encrypted_data_base64 = file.read()

    encrypted_data = base64.b64decode(encrypted_data_base64)
    decrypted_bytes = decrypt_data(fixed_key, encrypted_data)
    decrypted_embeddings[key] = np.frombuffer(decrypted_bytes, dtype=np.float32)
    assert np.allclose(
        combined_embeddings[key], decrypted_embeddings[key]
    ), f"Decryption failed for identity {key}"

print("Encryption and decryption completed successfully.")
