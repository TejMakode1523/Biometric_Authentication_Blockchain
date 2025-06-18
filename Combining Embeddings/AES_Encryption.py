import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Load the embeddings
fingerprint_embeddings = np.load("final_user_embeddings.npy", allow_pickle=True).item()
facial_embeddings = np.load("training_embeddings.npy", allow_pickle=True).item()

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

print(combined_embeddings["User_1"])
print(fingerprint_embeddings["User_1"])
print(facial_embeddings["User_1"])


# Save combined embeddings to a file
def save_combined_embeddings(embeddings, filename):
    np.save(filename, embeddings)


save_combined_embeddings(combined_embeddings, "combined_embeddings.npy")

print(combined_embeddings)


def derive_key_from_embedding(embedding):
    # Normalize the embedding (optional, depending on the nature of your embeddings)
    embedding = (embedding - np.min(embedding)) / (
        np.max(embedding) - np.min(embedding)
    )

    # Hash the embedding to create a 256-bit key
    embedding_bytes = embedding.tobytes()
    hash_obj = hashlib.sha256(embedding_bytes)
    key = hash_obj.digest()

    return key


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


print(combined_embeddings["User_1"])
# Encrypt the combined embeddings
encrypted_embeddings = {}
for key in combined_embeddings.keys():
    # Derive AES key from fingerprint embedding
    derived_key = derive_key_from_embedding(fingerprint_embeddings[key])
    # Encrypt the combined embedding
    combined_embedding_bytes = combined_embeddings[key].astype(np.float32).tobytes()
    encrypted_embeddings[key] = encrypt_data(derived_key, combined_embedding_bytes)

print(len(encrypted_embeddings["User_1"]))

# Decrypt and verify
decrypted_embeddings = {}
for key in encrypted_embeddings.keys():
    derived_key = derive_key_from_embedding(fingerprint_embeddings[key])
    decrypted_bytes = decrypt_data(derived_key, encrypted_embeddings[key])
    decrypted_embeddings[key] = np.frombuffer(decrypted_bytes, dtype=np.float32)
    assert np.allclose(
        combined_embeddings[key], decrypted_embeddings[key]
    ), f"Decryption failed for identity {key}"

print(decrypted_embeddings["User_1"])

print("Encryption and decryption completed successfully.")
