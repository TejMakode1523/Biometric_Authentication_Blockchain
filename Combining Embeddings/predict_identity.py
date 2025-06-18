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
face_image_path = "D:/pranav/NIT WARANGAL/Final Year Project/code/Models/Face_dataset_kaggle/User_1/Virat Kohli_0.jpg"
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
    "User_21",
    "User_22",
    "User_23",
    "User_24",
    "User_25",
    "User_26",
    "User_27",
    "User_28",
    "User_29",
    "User_3",
    "User_30",
    "User_31",
    "User_32",
    "User_33",
    "User_34",
    "User_35",
    "User_36",
    "User_37",
    "User_38",
    "User_39",
    "User_4",
    "User_40",
    "User_41",
    "User_42",
    "User_43",
    "User_44",
    "User_45",
    "User_46",
    "User_47",
    "User_48",
    "User_49",
    "User_5",
    "User_50",
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
if embedding_face is not None:
    print(f"Computed embedding: {embedding_face}")
else:
    print("No face detected in the image.")


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
input_image_path = "D:/pranav/NIT WARANGAL/Final Year Project/code/Models/Fingerprint_dataset_FVC2002DB_1/User_1/012_3_1.tif"
embeddings_fingerprint = calculate_embeddings(input_image_path)

print(f"Embeddings for the fingerprint image: {embeddings_fingerprint}")


def save_embeddings_to_file(face_embedding, fingerprint_embedding, output_file):
    # Concatenate the embeddings
    combined_embedding = np.concatenate((face_embedding, fingerprint_embedding), axis=1)
    print("Combined Embedding")
    print(combined_embedding)
    # Save the combined embedding to a file
    np.save(output_file, combined_embedding)
    print(f"Combined embedding saved to {output_file}")


if embedding_face is not None and embeddings_fingerprint is not None:
    # Ensure embeddings have the same dimensions
    if embedding_face.shape[1] == embeddings_fingerprint.shape[1]:
        save_embeddings_to_file(
            embeddings_fingerprint, embedding_face, "combined_embedding_result.npy"
        )
    else:
        print("Error: Embeddings have different dimensions and cannot be combined.")
else:
    print("One or both embeddings could not be computed.")


# The combined embedding are [[  0.           0.           0.           0.           0.
#     0.          38.746418     0.           0.           0.
#     0.           0.          50.424915     0.           0.
#     0.          55.198982    15.237132    19.426233     0.
#     0.           0.          64.72571     26.461205   113.880196
#     0.          44.462704     0.          28.016626    50.86522
#    51.55739      0.           0.           0.          54.65792
#    39.321922    37.361282     0.          15.215125     0.
#     0.           0.           0.          56.649307     0.
#    20.035604    19.1071       0.           0.          42.443703
#     0.          19.488985    45.346256    37.032055     0.
#    28.04238      0.           0.           0.          47.799805
#    45.0813       0.           0.          38.980625    78.19203
#    21.492907     0.           0.          37.812557    15.619273
#     0.           0.          42.53882     47.843613     0.
#    22.833025    40.51766     56.639187     0.          37.6124
#    28.04572     29.504053     0.           0.           0.
#     0.          43.694534    44.654182     0.          48.96814
#     0.          35.07469      0.           0.          48.54226
#     0.           0.           0.           0.           0.
#     0.           0.          54.993393     0.          86.92581
#    34.819946    18.101995     0.           0.           0.
#     0.           0.           0.           0.          45.476837
#     0.           0.           0.           0.           0.
#     0.          51.934803    27.620283     0.          44.287945
#     0.          26.077484     0.           0.           5.771181
#     0.           1.8470881    0.           1.1735198    2.8730903
#     0.69252604   2.113049     0.           0.           1.1649189
#     0.           0.           3.5231936    3.5776165    0.4110091
#     0.           6.9005957    1.0153577    3.535634     0.
#     0.           0.8989744    0.           0.5400232    1.4048356
#     0.92227304   0.68267494   0.8050536    0.62037516   0.
#     0.           0.           0.           2.1908565    0.
#     0.16851263   9.182893     0.57031906   3.791823     1.9494907
#     1.1075777    0.           0.           3.904309     0.
#     0.           0.           1.9020615    0.           0.4760946
#     1.6447556    0.81505084   4.447091     0.           0.
#     1.6430007    0.           0.           4.198983     0.
#     0.           6.73683      0.           0.           1.7022176
#     0.           0.           0.           0.           6.5413027
#     0.           0.           0.           0.5195949    2.120852
#     7.020696     0.           0.           0.           0.
#     1.2057887    1.9613131    7.542211     5.860535     0.
#     0.3309877    0.647304     2.863414     2.3509912    0.
#     0.           0.           4.440951     0.           0.48443496
#     7.8720675    0.           2.8158152    8.265088     0.
#     2.5056255    3.4465652    0.           0.14142253   1.3803239
#     0.           0.           0.           0.           0.
#     1.3397596    0.           1.0983853    2.0621011    0.
#     0.           0.           1.5326755    1.6034675    0.
#     0.           0.           5.4797163    0.           3.7826204
#     0.        ]]


# Combined Embedding
# [[0.0000000e+00 7.0608727e+01 0.0000000e+00 4.7766415e+01 0.0000000e+00
#   3.9134563e+01 3.7986916e+01 0.0000000e+00 7.7043152e+01 0.0000000e+00
#   0.0000000e+00 7.5538216e+01 6.6911453e+01 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 3.3318687e+01 6.0490837e+01 2.2822380e+01
#   0.0000000e+00 7.7192780e+01 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   3.8890095e+01 2.5207546e+01 1.5964730e+01 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 6.7173454e+01 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 6.3727047e+01 1.5761047e+02 0.0000000e+00 2.7619375e+01
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 6.3740295e+01
#   0.0000000e+00 0.0000000e+00 6.2948009e+01 0.0000000e+00 6.5015572e+01
#   0.0000000e+00 7.9367661e+01 0.0000000e+00 5.9349476e+01 0.0000000e+00
#   0.0000000e+00 8.6784523e+01 7.1409409e+01 0.0000000e+00 4.5397915e+01
#   0.0000000e+00 6.0636948e+01 0.0000000e+00 5.6278236e+01 0.0000000e+00
#   5.9317749e+01 0.0000000e+00 8.2871231e+01 0.0000000e+00 3.4100121e+01
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 3.2995411e+01 0.0000000e+00 0.0000000e+00
#   4.0908485e+01 0.0000000e+00 4.7368824e+01 0.0000000e+00 0.0000000e+00
#   6.6759354e+01 8.8461655e+01 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   4.3460106e+01 7.4462662e+01 0.0000000e+00 7.6292091e+01 0.0000000e+00
#   4.1588055e+01 0.0000000e+00 0.0000000e+00 5.4205730e+01 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 5.3336979e+01 0.0000000e+00 5.8764454e+01
#   0.0000000e+00 6.0463566e+01 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 5.7711811e+00
#   0.0000000e+00 1.8470881e+00 0.0000000e+00 1.1735198e+00 2.8730903e+00
#   6.9252604e-01 2.1130490e+00 0.0000000e+00 0.0000000e+00 1.1649189e+00
#   0.0000000e+00 0.0000000e+00 3.5231936e+00 3.5776165e+00 4.1100910e-01
#   0.0000000e+00 6.9005957e+00 1.0153577e+00 3.5356340e+00 0.0000000e+00
#   0.0000000e+00 8.9897442e-01 0.0000000e+00 5.4002321e-01 1.4048356e+00
#   9.2227304e-01 6.8267494e-01 8.0505359e-01 6.2037516e-01 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 2.1908565e+00 0.0000000e+00
#   1.6851263e-01 9.1828928e+00 5.7031906e-01 3.7918229e+00 1.9494907e+00
#   1.1075777e+00 0.0000000e+00 0.0000000e+00 3.9043090e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 1.9020615e+00 0.0000000e+00 4.7609460e-01
#   1.6447556e+00 8.1505084e-01 4.4470911e+00 0.0000000e+00 0.0000000e+00
#   1.6430007e+00 0.0000000e+00 0.0000000e+00 4.1989832e+00 0.0000000e+00
#   0.0000000e+00 6.7368302e+00 0.0000000e+00 0.0000000e+00 1.7022176e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 6.5413027e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 5.1959491e-01 2.1208520e+00
#   7.0206962e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   1.2057887e+00 1.9613131e+00 7.5422111e+00 5.8605351e+00 0.0000000e+00
#   3.3098769e-01 6.4730400e-01 2.8634140e+00 2.3509912e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 4.4409509e+00 0.0000000e+00 4.8443496e-01
#   7.8720675e+00 0.0000000e+00 2.8158152e+00 8.2650881e+00 0.0000000e+00
#   2.5056255e+00 3.4465652e+00 0.0000000e+00 1.4142253e-01 1.3803239e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   1.3397596e+00 0.0000000e+00 1.0983853e+00 2.0621011e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 1.5326755e+00 1.6034675e+00 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 5.4797163e+00 0.0000000e+00 3.7826204e+00
#   0.0000000e+00]]
