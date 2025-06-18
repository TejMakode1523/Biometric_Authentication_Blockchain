import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from keras.models import load_model
from web3 import Web3
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import requests
import base64

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load Haar Cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load the models
model_face = load_model("models/fine_tuned_arcface_model_with_haar.h5", compile=False)
feature_extractor = tf.keras.Model(
    inputs=model_face.input, outputs=model_face.layers[-2].output
)
siamese_model = load_model("models/siamese_model.h5")
# Extract the base network to calculate embeddings
base_network = siamese_model.get_layer(index=2)


# Web3 setup
# Connect to Ethereum node
# Connect to Ethereum node
# Connect to Ethereum node
infura_url = "http://127.0.0.1:8545"  # Replace with your Infura URL or local node URL
web3 = Web3(Web3.HTTPProvider(infura_url))

# Check connection
if web3.is_connected():
    print("Connected to Ethereum node")
else:
    print("Failed to connect to Ethereum node")

# Smart contract details
contract_address = "0xcd8c600e2ebf4ab6b8a08d9aae2aeb9eb0a94146"  # Replace with your deployed contract address
contract_address = Web3.to_checksum_address(contract_address)

contract_abi = [
    {
        "inputs": [
            {"internalType": "string", "name": "userId", "type": "string"},
            {"internalType": "string", "name": "hashValue", "type": "string"},
        ],
        "name": "storeUserHash",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "userId", "type": "string"}],
        "name": "getUserHash",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Set up contract
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Wallet details
wallet_address = (
    "0x2a8De603a1BA00A3732D9F1F89425d5Aa4Cd9b09"  # Replace with your wallet address
)
wallet_address = Web3.to_checksum_address(wallet_address)
private_key = "0xc4d8b3b1fddcbd899ee5e7e9ae5453321accc325ed01a76cd5f20b6ac52ec046"  # Replace with your private key


# Function to verify if a CID exists on the blockchain
def extract_cid(user_id):
    result = contract.functions.getUserHash(user_id).call()
    if result:
        print(f"The {user_id} has hash value {result}")
    else:
        print(f"The user does not exist on the blockchain.")
    return result


# Preprocessing function
# Function to detect and crop face
def detect_and_crop_face(image_path, target_size=(112, 112)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"No face detected in {image_path}. Skipping.")
        return None

    # Crop the first detected face
    x, y, w, h = faces[0]
    cropped_face = img[y : y + h, x : x + w]
    cropped_face = cv2.resize(cropped_face, target_size)
    cropped_face = cropped_face / 255.0  # Normalize
    return cropped_face


# Function to predict identity
def predict_identity_face(image_path):
    cropped_face = detect_and_crop_face(image_path)
    if cropped_face is None:
        return "No face detected."

    cropped_face = np.expand_dims(cropped_face, axis=0)  # Add batch dimension
    prediction = model_face.predict(cropped_face)

    # You may need to process the prediction further based on your specific model's output
    identity = np.argmax(prediction)  # Example for classification model
    return identity


# Function to compute embedding
def compute_embedding(image_path):
    cropped_face = detect_and_crop_face(image_path)
    if cropped_face is None:
        return None

    cropped_face = np.expand_dims(cropped_face, axis=0)  # Add batch dimension
    embedding = feature_extractor.predict(cropped_face)

    return embedding


# Function to preprocess the input image
def preprocess_fingerprint_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = img.reshape(-1, 128, 128, 1)  # Reshape for the model
    return img


# Function to calculate embeddings for an input fingerprint image
def calculate_embeddings(input_image_path):
    input_image = preprocess_fingerprint_image(input_image_path)

    # Calculate embeddings using the base network
    embeddings = base_network.predict(input_image)

    return embeddings


def decrypt_data(key, encrypted_data):
    iv = encrypted_data[: AES.block_size]
    ciphertext = encrypted_data[AES.block_size :]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return plaintext


# Function to calculate the Euclidean distance between two arrays
def calculate_euclidean_distance(array1, array2):
    return np.linalg.norm(array1 - array2)


def combine_embedding(face_embedding, fingerprint_embedding):
    # Concatenate the embeddings
    combined_embedding = np.concatenate((face_embedding, fingerprint_embedding), axis=1)
    return combined_embedding


class_labels_face = [
    "User_1",
    "User_10",
    "User_11",
    "User_12",
    "User_13",
    "User_14",
    "User_15",
    "User_16",
    "User_17",
    "User_18",
    "User_19",
    "User_2",
    "User_20",
    "User_3",
    "User_4",
    "User_5",
    "User_6",
    "User_7",
    "User_8",
    "User_9",
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img1 = request.files["image1"]
        img2 = request.files["image2"]
        img1_path = os.path.join(app.config["UPLOAD_FOLDER"], img1.filename)
        img2_path = os.path.join(app.config["UPLOAD_FOLDER"], img2.filename)
        img1.save(img1_path)
        img2.save(img2_path)

        identity_face = predict_identity_face(img1_path)
        embedding1 = compute_embedding(img1_path)
        embedding2 = calculate_embeddings(img2_path)

        if embedding1 is not None and embedding2 is not None:
            # Ensure embeddings have the same dimensions
            if embedding1.shape[1] == embedding2.shape[1]:
                combined_embedding = combine_embedding(embedding2, embedding1)
                print(f"The combined embedding are {combined_embedding}")
            else:
                print(
                    "Error: Embeddings have different dimensions and cannot be combined."
                )
                return render_template(
                    "index.html", result="No face detected in one of the images."
                )
        else:
            print("One or both embeddings could not be computed.")
            return render_template(
                "index.html", result="No face detected in one of the images."
            )

        print(f"The identity predicted is {class_labels_face[identity_face]}")
        cid = extract_cid(class_labels_face[identity_face])
        print(f"The cid is {cid}")

        # Pinata gateway URL
        gateway_url = f"https://gateway.pinata.cloud/ipfs/{cid}"

        # Fetch the file content
        response = requests.get(gateway_url)

        if response.status_code == 200:
            # Load the file content
            file_content = response.text  # or response.content for binary files
            print("File content loaded successfully:")
            print(file_content)
        else:
            print(
                f"Failed to load file content: {response.status_code}, {response.text}"
            )

        fixed_key = b"ThisIsA32ByteFixedKeyForAES256!!"

        encrypted_data = base64.b64decode(file_content)
        decrypted_bytes = decrypt_data(fixed_key, encrypted_data)
        decrypted_embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)

        print("The decrypted Embeddings are as follows")
        print(decrypted_embedding)

        distance = calculate_euclidean_distance(decrypted_embedding, combined_embedding)
        print(
            f"Euclidean distance between decrypted embedding and additional array : {distance}"
        )
        threshold = 30  # Set based on experiments

        if distance < threshold:
            result = f"Authenticated {class_labels_face[identity_face]}"
        else:
            result = "Invalid User"
        return render_template("index.html", result=result)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
