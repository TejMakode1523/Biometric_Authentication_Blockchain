# 🔐 Multi-Biometric Authentication Using Blockchain

## 📌 Overview
This project presents a **secure and decentralized multi-biometric authentication system** using **blockchain and IPFS**. It integrates **face recognition** and **fingerprint recognition** to enhance authentication accuracy and security. The biometric feature embeddings are **AES-256 encrypted** before storage, and the encrypted data is uploaded to **IPFS**, generating a unique **CID (Content Identifier)**. 

The **CID is stored on a smart contract** deployed on a **private blockchain (Ganache)**, ensuring **immutability and security**. During authentication, the CID is retrieved, decrypted, and matched using **Euclidean distance-based verification**. This approach enhances **data security, privacy, and resistance to tampering**.

---

## 📌 Features
✔ **Multi-Modal Biometric Authentication** (Face + Fingerprint)  
✔ **Blockchain-Based Secure Storage** (Smart Contracts on Ethereum)  
✔ **Decentralized Storage using IPFS**  
✔ **AES-256 Encryption** (Ensuring Biometric Data Security)  
✔ **Tamper-Proof Identity Verification**  
✔ **Integration with Fuzzy Vault for Additional Security**  
✔ **High Accuracy and Low Error Rates**  

---

## 📌 System Workflow
### 🟢 Enrollment Phase
1. **Acquisition of Biometric Data**: Face and fingerprint images are captured.  
2. **Pre-Processing**: Standard methods and **Haar Cascade classifier** are used to extract faces from images.  
3. **Feature Extraction**:  
   - **Face Recognition Model** (ArcFace) extracts 128-dimensional embeddings.  
   - **Fingerprint Recognition Model** extracts 128-dimensional embeddings.  
4. **Concatenation and Encryption**: The feature vectors are concatenated and **AES-256 encrypted**.  
5. **Storage on IPFS**: The encrypted embeddings are uploaded to **IPFS**, generating a **CID**.  
6. **Smart Contract Interaction**: The CID is stored on the blockchain for **tamper-proof security**.

### 🟢 Authentication Phase
1. **Capture New Biometric Data**: User provides live face and fingerprint samples.  
2. **Feature Extraction**: The new embeddings are extracted similarly to the enrollment phase.  
3. **CID Retrieval from Blockchain**: The **smart contract** is queried to retrieve the CID.  
4. **Decryption & Matching**: The retrieved encrypted embeddings are **AES-decrypted**, and **Euclidean distance** is computed for verification.  
5. **Decision Making**: If the **distance is below the threshold**, authentication is successful.  

---

## 📌 Technologies Used
- **Programming Language**: Python  
- **Blockchain**: Ethereum, Ganache, Solidity, Truffle  
- **Biometric Models**:  
  - **Face Recognition**: ArcFace  
  - **Fingerprint Recognition**: Saimese Network  
- **Cryptography**: AES-256 Encryption  
- **Storage**: Pinata IPFS (InterPlanetary File System)  
- **Smart Contracts**: Solidity (Executed via Remix IDE)  
- **Libraries**: OpenCV, NumPy, Tensorflow, PyCryptodome, Web3, Truffle  
- ArcFace Model Download Link : https://huggingface.co/felixrosberg/ArcFace/blob/main/ArcFacePerceptual-Res50.h5
---

## 📌 Installation Guide
### 🔹 Prerequisites
Ensure the following are installed on your system:
- **Python 3.8+**
- **Ganache** (For local blockchain setup)
- **Truffle** (For smart contract deployment)
- **MetaMask** (For Ethereum wallet integration)

### 🔹 Steps to Set Up & Run
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-repo/multi-biometric-auth-blockchain.git
cd multi-biometric-auth-blockchain
```
2️⃣ Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
3️⃣ Install Dependencies
```bash
pip install opencv-pyhton
pip install numpy
pip install tensorflow==2.14
pip install web3
pip install PyCrytodome
```
4️⃣ Start Ganache Blockchain

Open Ganache and create a new workspace.
Set the network port to 8545 .

5️⃣ Compile & Deploy Smart Contracts
```
truffle compile
truffle migrate --reset
```
Instead use Remix IDE to compile and Deploy.

6️⃣ Run the Authentication System
```
python app.py
```
---

## 📌 Experimental Setup
### 🟢 System Configuration
- Processor: Intel Core i5 (1.70 GHz)
- RAM: 16 GB
- Blockchain Network: Ethereum (Local Ganache Network)
- Truffle Framework: Used for smart contract execution
### 🟢 Dataset
- Total Identities: 20
- Dataset Links:
   - Fingerprint : https://drive.google.com/drive/folders/1QAzNU6qXUFgx3YVvgmRuvdRhtSVNmcFn?usp=sharing
   - Face : https://drive.google.com/drive/folders/1iSNyL23AR_al_gfesuB6TmHp-GlH78aE?usp=sharing

---

## 📌 Results & Discussion
### ✅ Biometric Model Accuracy
- Face Recognition Accuracy: 93%
- Fingerprint Recognition Accuracy: 98%
- Multi-Modal System Equal Error Rate (EER): 0.0297
### ✅ Impact of Multi-Modal Biometric Authentication
- False Acceptance Rate (FAR): 2.67%
- False Rejection Rate (FRR): 3.28%
### ✅ Security & Performance Metrics
- AES Encryption & Decryption Time: 1.5 ms per embedding
- CID Retrieval Time from Blockchain: 28 ms
- This demonstrates that combining face and fingerprint biometrics significantly improves security and accuracy. The blockchain-based storage of CIDs prevents tampering, ensuring secure identity verification.

---

## 📌 Comparison with Existing Systems
- Feature	Traditional Biometric Systems	Proposed Multi-Biometric Blockchain System
- Authentication Method	Single-Modality (Face/Fingerprint)	Multi-Modal (Face + Fingerprint)
- Storage Type	Centralized Databases	Decentralized (IPFS + Blockchain)
- Encryption	Basic Hashing Methods	AES-256 Encryption
- Security	Vulnerable to Tampering	Tamper-Proof (Blockchain)
- Accuracy	~85-90%	93-98% (Multi-Modal Fusion)

---

## 📌 Future Scope
- 🚀 Deployment on Public Blockchains (Ethereum Mainnet, Hyperledger)
- 🚀 Integration with Cloud Storage for scalable biometric data management
- 🚀 Improved Encryption (Homomorphic Encryption for privacy-preserving authentication)
- 🚀 Optimized Smart Contracts for lower gas costs and improved efficiency
