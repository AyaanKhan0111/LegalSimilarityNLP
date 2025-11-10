# Deep Learning Assignment 2 — Legal Clause Similarity (CS-452)

**Author:** Ayaan Khan | DS-D | 22I-2066  
**Course:** Deep Learning for Perception  
**Instructor:** *[Add instructor name if required]*  
**Institute:** FAST NUCES  

---

## Overview

This project implements **two baseline deep learning architectures** to identify **semantic similarity between legal clauses**.  
Legal clauses are written in formal, domain-specific language where the same principle can be expressed in multiple ways.  
The models aim to detect when two clauses **mean the same thing**, even if worded differently.

---

## Dataset

Dataset used: [Legal Clause Dataset — Kaggle](https://www.kaggle.com/datasets/bahushruth/legalclausedataset)

Each CSV file corresponds to a **clause category** (e.g., *acceleration*, *access-to-information*, *accounting-terms*).  
Positive and negative pairs were generated for training:

- **Positive pairs (1):** Clauses from the same category  
- **Negative pairs (0):** Clauses from different categories  

Balanced dataset (1:1 ratio across all splits).

| Split | Similar (1) | Dissimilar (0) | Total Samples | Ratio |
|--------|--------------|----------------|----------------|--------|
| Train | 227,808 | 227,807 | 455,615 | 50.00% |
| Validation | 40,201 | 40,202 | 80,403 | 50.00% |
| Test | 47,296 | 47,296 | 94,592 | 50.00% |

---

## Data Preprocessing

- Text cleaning: lowercasing, punctuation removal  
- Tokenization: whitespace-based  
- Vocabulary size: 30,000 tokens  
- Sequence length: 200 tokens (padded/truncated)  
- Split: 70% train / 15% val / 15% test (stratified)

---

## Model Architectures

### Model 1 — Siamese BiLSTM
- Embedding Dim: 128  
- Hidden Dim: 128  
- Layers: 1 (Bidirectional)  
- Dropout: 0.2  
- Classifier: Concatenation of absolute difference + element-wise product  
- Loss: Binary Cross-Entropy with Logits  
- Optimizer: Adam (lr=1e-3)  
- Epochs: 10  
- Batch Size: 64  

**Rationale:**  
Captures sequential and contextual dependencies between words using bidirectional recurrence.  
Encourages meaningful embeddings for semantic similarity.

---

### Model 2 — Self-Attention Encoder
- Embedding Dim: 128  
- Attention Heads: 4  
- Feedforward Hidden Size: 256  
- Dropout: 0.2  
- Loss/Optimizer: Same as above  

**Rationale:**  
Leverages self-attention to identify contextually important words.  
Better handles long and complex legal clauses by focusing on key tokens like *termination*, *liability*, *warranty*.

---

## Training Setup

- Framework: **PyTorch**
- Epochs: **10**
- Batch Size: **64**
- Device: GPU/CPU (auto-detected)
- Checkpoints saved: `siamese_bilstm_best.pt`, `attn_encoder_best.pt`

**Command-line run example (Colab/Jupyter):**
```bash
!python DL_Assignment2_code.ipynb
