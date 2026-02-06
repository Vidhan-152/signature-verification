# ✍️ Signature Verification using Siamese Neural Networks

## 📌 Project Overview

This project implements a **writer-independent offline signature verification system** using **Deep Learning**.  
Given two signature images, the system determines whether they belong to the **same person (Genuine)** or **different people (Forged)**.

Unlike traditional classification-based approaches, this system is designed as a **verification problem**, using **metric learning** and a **Siamese Neural Network** trained with **contrastive loss**.

The trained model is deployed as a **web application** using **Streamlit**, allowing real-time verification without retraining.

Link : https://signature-verification-project.streamlit.app/

---

## 🎯 Problem Statement

Signature verification is a biometric authentication problem where the goal is to verify a person’s identity based on their handwritten signature.

Key challenges:
- High intra-personal variation (same person signs differently)
- Skilled forgeries
- Generalization to unseen users

This project focuses on **writer-independent verification**, meaning the model can verify signatures of users **not seen during training**.

---

## 🧠 Core Concepts Used

### Siamese Neural Network

A **Siamese Network** consists of two identical neural networks that share weights.

**Architecture logic:**
Signature 1 ──► CNN ──► Embedding f1
Signature 2 ──► CNN ──► Embedding f2

- Both images pass through the **same CNN**
- The CNN learns to extract **writing-style features**
- Output is a **fixed-length embedding vector**

  <img width="838" height="297" alt="image" src="https://github.com/user-attachments/assets/c476aa2b-6960-44e8-bccd-7579abed433f" />


---

### Feature Embeddings

Instead of predicting labels directly, the network maps each signature into a **128-dimensional embedding space**.

- Similar signatures → embeddings close together
- Different signatures → embeddings far apart

This enables similarity-based decision making.

---

### Contrastive Loss (Metric Learning)

To train the embedding space, **contrastive loss** is used.

**Intuition:**
- Genuine pairs → pull embeddings closer
- Forged pairs → push embeddings apart (by a margin)

**Loss function:**
L = y * D² + (1 - y) * max(margin - D, 0)²

Where:
- `D` = Euclidean distance between embeddings
- `y` = 1 for genuine, 0 for forged
- `margin` = minimum required separation for forged pairs

This loss directly optimizes **similarity geometry**, making it ideal for verification tasks.

<img width="550" height="358" alt="image" src="https://github.com/user-attachments/assets/ec2f1b49-339b-460e-8ef6-db930653e783" />


---

### Distance-Based Decision

After training:
- The network outputs **embeddings**, not decisions
- Similarity is measured using **Euclidean distance**

distance = ||f1 - f2||


Smaller distance → more similar signatures.

---

### Threshold-Based Verification

A **threshold** is used to convert distance into a binary decision:

if distance ≤ threshold → Genuine
else → Forged


The threshold is:
- **Not learned during training**
- Selected using a **validation set**
- Tuned to balance false acceptance and false rejection

This separation allows flexible deployment in different security scenarios.

---
