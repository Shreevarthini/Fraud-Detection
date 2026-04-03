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

https://huggingface.co/spaces/Shreevarthini/Fraud-Detection-CC

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

### Understanding the Features (V1 - V28)
If you are visiting this app for the first time, you might notice the features are labeled V1 through V28. Here is why:

Privacy Protection: The original data contains sensitive credit card information (like merchant names, locations, and personal habits). To comply with privacy laws, the researchers used PCA (Principal Component Analysis).

What is PCA?: It is a dimensionality reduction technique that transforms raw data into a new set of variables. While we can no longer see the "original" names (like "Merchant Category"), the mathematical patterns and correlations of fraud are preserved.

Key Indicators: In this model, features like V14, V17, and V12 are often the most significant "signals" for detecting fraudulent behavior.

Included Raw Features: Only Time and Amount remained in their original form and were scaled during the preprocessing phase.

### DATASET 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
