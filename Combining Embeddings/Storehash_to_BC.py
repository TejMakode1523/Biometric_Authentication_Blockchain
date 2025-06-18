# storing the files in cloud #
import os
import time
import requests

# Pinata API credentials
pinata_api_key = "c946eaad217c396698fd"
pinata_secret_api_key = (
    "5d7107d45ea5dc48f5720adeb9f6039e716b2803746bd09c6ffe6e958c445998"
)

# Pinata API base URL
pinata_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

# List of files to upload
file_paths = [
    f"User_{i}_encrypted.txt" for i in range(1, 51)
]  # Replace with your actual file paths

# Headers for Pinata API request
headers = {
    "pinata_api_key": pinata_api_key,
    "pinata_secret_api_key": pinata_secret_api_key,
}

# Array to store CIDs
cids = []

# Loop through each file and upload to Pinata
for file_path in file_paths:
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(pinata_url, headers=headers, files=files)
        if response.status_code == 200:
            result = response.json()
            ipfs_hash = result["IpfsHash"]
            print(f"File {file_path} uploaded successfully! CID: {ipfs_hash}")
            cids.append(ipfs_hash)
        else:
            print(f"Error uploading file {file_path}: {response.text}")

# Print all CIDs
print("All CIDs Uploaded Successfully !!")


# ----------------------------------------------------#
# Storing the cid in blockchain #
# ----------------------------------------------------#

user_ids = [
    "User_1",
    "User_2",
    "User_3",
    "User_4",
    "User_5",
    "User_6",
    "User_7",
    "User_8",
    "User_9",
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
    "User_50",
]

from web3 import Web3

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


# Function to store CID in blockchain
def store_cid(user_id, cid, total_time_taken):
    # Build transaction
    nonce = web3.eth.get_transaction_count(wallet_address)
    txn = contract.functions.storeUserHash(user_id, cid).build_transaction(
        {
            "chainId": 1337,  # Mainnet, replace with the appropriate chain ID if using a testnet
            "gas": 2000000,
            "gasPrice": web3.to_wei("20", "gwei"),
            "nonce": nonce,
        }
    )

    # Sign transaction
    signed_txn = web3.eth.account.sign_transaction(txn, private_key=private_key)

    # Record the start time
    start_time = time.time()

    # Send transaction
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"Transaction hash: {web3.to_hex(tx_hash)}")

    # Wait for transaction receipt
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Transaction receipt: {tx_receipt}")

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    total_time_taken = total_time_taken + execution_time
    print(f"The exection time for {user_id} is {execution_time}")


total_time_taken = 0
# Storing the CID
count = 0
for cid in cids:
    store_cid(user_ids[count], cid, total_time_taken)
    count = count + 1

print(f"The total time taken to upload the hash to blockchain is {total_time_taken}")
