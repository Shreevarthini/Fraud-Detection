---
title: Credit Card Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection System

This project demonstrates a **Real-Time Fraud Detection Engine** built with XGBoost. It addresses the challenge of extreme class imbalance (0.17% fraud rate) using **SMOTE** (Synthetic Minority Over-sampling Technique).

### Features
* **XGBoost Classifier:** Optimized for **Recall** to minimize missed fraudulent transactions.
* **SMOTE Integration:** Balanced the training set to ensure the model learns fraud patterns effectively.
* **Interactive Dashboard:** Allows users to simulate transactions and adjust "V-features" (PCA components).

### Technical Stack
* **Language:** Python
* **ML Libraries:** XGBoost, Scikit-learn, Imbalanced-learn
* **Deployment:** Streamlit & Hugging Face Spaces

### How to Test
1. Use the **"Suspicious Pattern"** button in the sidebar to load a known risky profile.
2. Click **"Analyze Transaction"** to see the probability score and recommended action.

### DATASET 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
