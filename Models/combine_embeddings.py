import numpy as np
import hashlib
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

print(combined_embeddings["User_1"])
print(fingerprint_embeddings["User_1"])
print(facial_embeddings["User_1"])


# Save combined embeddings to a file
def save_combined_embeddings(embeddings, filename):
    np.save(filename, embeddings)


save_combined_embeddings(combined_embeddings, "combined_embeddings.npy")
