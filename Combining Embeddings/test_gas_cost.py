import os
import time
import requests
from web3 import Web3

# Pinata API credentials
pinata_api_key = "c946eaad217c396698fd"
pinata_secret_api_key = (
    "5d7107d45ea5dc48f5720adeb9f6039e716b2803746bd09c6ffe6e958c445998"
)

# Pinata API base URL
pinata_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

# List of files to upload
file_paths = [f"User_{i}_encrypted.txt" for i in range(1, 21)]

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

print("All CIDs Uploaded Successfully !!")

# Ethereum blockchain connection
infura_url = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(infura_url))

if web3.is_connected():
    print("Connected to Ethereum node")
else:
    print("Failed to connect to Ethereum node")

contract_address = "0x9bced8592e48334a89a3181852c788a1046188e3"
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

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

wallet_address = "0x8Cd1A918282B6C4c102428BEeBd7596Dc41D4F2A"
wallet_address = Web3.to_checksum_address(wallet_address)
private_key = "0x1b8791f64ee836a67ddbe2d27148a5388a6e5726cb0262b2df0031e10bca6e8e"

total_gas_used = 0
total_gas_cost = 0
total_time_taken = 0


def store_cid(user_id, cid):
    global total_gas_used, total_gas_cost, total_time_taken
    nonce = web3.eth.get_transaction_count(wallet_address)
    txn = contract.functions.storeUserHash(user_id, cid).build_transaction(
        {
            "chainId": 1337,
            "gas": 2000000,
            "gasPrice": web3.to_wei("20", "gwei"),
            "nonce": nonce,
        }
    )

    signed_txn = web3.eth.account.sign_transaction(txn, private_key=private_key)
    start_time = time.time()
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"Transaction hash: {web3.to_hex(tx_hash)}")

    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    gas_used = tx_receipt.gasUsed
    gas_price = web3.to_wei("20", "gwei")
    gas_cost = gas_used * gas_price
    end_time = time.time()

    execution_time = end_time - start_time
    total_time_taken += execution_time
    total_gas_used += gas_used
    total_gas_cost += gas_cost

    print(f"Gas used for {user_id}: {gas_used}")
    print(f"Gas cost for {user_id}: {web3.from_wei(gas_cost, 'ether')} ETH")
    print(f"Execution time for {user_id}: {execution_time} seconds")


for i, cid in enumerate(cids):
    store_cid(f"User_{i+1}", cid)

print(f"Total gas used: {total_gas_used}")
print(f"Total gas cost: {web3.from_wei(total_gas_cost, 'ether')} ETH")
print(f"Total time taken: {total_time_taken} seconds")
