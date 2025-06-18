# Embedding generation of the User#
# Works in Google Colab only#

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model

# Load Haar Cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)


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


# Load the fine-tuned ArcFace model
model_face = load_model("fine_tuned_arcface_model_with_haar.h5", compile=False)

feature_extractor = tf.keras.Model(
    inputs=model_face.input, outputs=model_face.layers[-2].output
)


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


# Example usage
face_image_path = "Face_dataset_kaggle/User_11/Claire Holt_0.jpg"
identity_face = predict_identity_face(face_image_path)
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
if identity_face != "No face detected.":
    print(
        f"Predicted identity: {identity_face} name: {class_labels_face[identity_face]}"
    )
else:
    print("Authentication failed !!")

embedding_face = compute_embedding(face_image_path)
# if embedding_face is not None:
#     print(f"Computed embedding: {embedding_face}")
# else:
#     print("No face detected in the image.")


# Function to preprocess the input image
def preprocess_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = img.reshape(-1, 128, 128, 1)  # Reshape for the model
    return img


# Load the trained Siamese model
siamese_model = load_model("siamese_model.h5")

# Extract the base network to calculate embeddings
base_network = siamese_model.get_layer(index=2)


# Function to calculate embeddings for an input fingerprint image
def calculate_embeddings(input_image_path):
    input_image = preprocess_image(input_image_path)

    # Calculate embeddings using the base network
    embeddings = base_network.predict(input_image)

    return embeddings


# Example usage
input_image_path = "Fingerprint_dataset_FVC2002DB_1/User_1/012_3_1.tif"
embeddings_fingerprint = calculate_embeddings(input_image_path)

# print(f"Embeddings for the fingerprint image: {embeddings_fingerprint}")
