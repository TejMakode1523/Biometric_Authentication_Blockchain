from web3 import Web3
import requests
import numpy as np
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Connect to Ethereum node
infura_url = "http://127.0.0.1:8545"  # Replace with your Infura URL or local node URL
web3 = Web3(Web3.HTTPProvider(infura_url))

# Check connection
if web3.is_connected():
    print("Connected to Ethereum node")
else:
    print("Failed to connect to Ethereum node")

# Smart contract details
contract_address = "0x89c0ed886f1afa6a052638cb1b1cf3644aab1a84"  # Replace with your deployed contract address
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
    "0xdA8c5124b9807b6AF94BE680Eb38CA48c486cdFC"  # Replace with your wallet address
)
wallet_address = Web3.to_checksum_address(wallet_address)
private_key = "0x81a0dd6bc79c972f51aa7bf1f11b71540546f25f4bc49e3112d8484748ceb7d4"  # Replace with your private key


# Function to verify if a CID exists on the blockchain
def extract_cid(user_id):
    result = contract.functions.getUserHash(user_id).call()
    if result:
        print(f"The {user_id} has hash value {result}")
    else:
        print(f"The user does not exist on the blockchain.")
    return result


# Example: Verify a CID
user_id = "User_1"  # Replace with the CID you want to verify
cid = extract_cid(user_id)

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
    print(f"Failed to load file content: {response.status_code}, {response.text}")

# Decryption of the file content
# Use a fixed key for encryption (must be 32 bytes for AES-256)
fixed_key = b"ThisIsA32ByteFixedKeyForAES256!!"


def decrypt_data(key, encrypted_data):
    iv = encrypted_data[: AES.block_size]
    ciphertext = encrypted_data[AES.block_size :]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return plaintext


encrypted_data = base64.b64decode(file_content)
decrypted_bytes = decrypt_data(fixed_key, encrypted_data)
decrypted_embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)

print("The decrypted Embeddings are as follows")
print(decrypted_embedding)


# Function to load the additional array from file (same size as decrypted data)
def load_additional_array(filename):
    return np.load(filename)


# Function to calculate the Euclidean distance between two arrays
def calculate_euclidean_distance(array1, array2):
    return np.linalg.norm(array1 - array2)


# Load the additional array
additional_array = load_additional_array(
    "combined_embedding_result.npy"
)  # Replace with actual filename

distance = calculate_euclidean_distance(decrypted_embedding, additional_array)
print(
    f"Euclidean distance between decrypted embedding and additional array : {distance}"
)

threshold = 50  # find ideal threshold by experimenting

if distance < threshold:
    print(f"{user_id} Authenticated !!")
else:
    print("Invlaid User !!")
