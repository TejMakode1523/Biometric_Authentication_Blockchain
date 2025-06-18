import os
import requests

# Pinata API credentials
pinata_api_key = "c946eaad217c396698fd"
pinata_secret_api_key = (
    "5d7107d45ea5dc48f5720adeb9f6039e716b2803746bd09c6ffe6e958c445998"
)

# Pinata API base URL
pinata_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

# The file to upload
file_path = "User_3_encrypted.txt"

# Headers for Pinata API request
headers = {
    "pinata_api_key": pinata_api_key,
    "pinata_secret_api_key": pinata_secret_api_key,
}

# Open the file to upload
with open(file_path, "rb") as file:
    files = {"file": file}

    # Upload the file to Pinata
    response = requests.post(pinata_url, headers=headers, files=files)

    if response.status_code == 200:
        # If the request was successful, print the CID (IPFS hash)
        result = response.json()
        ipfs_hash = result["IpfsHash"]
        print(f"File uploaded successfully! CID: {ipfs_hash}")

        # Access the file using the Pinata gateway
        gateway = "https://gateway.pinata.cloud/ipfs/"
        print(requests.get(url=gateway + ipfs_hash).text)
        print(gateway + ipfs_hash)
    else:
        print(f"Error uploading file: {response.text}")

# Additional actions you can perform:
# Get a list of pinned files, job status, etc. You would need additional endpoints for this.
