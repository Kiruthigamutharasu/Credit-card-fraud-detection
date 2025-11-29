# Credit Card Fraud Detection — ANN & Autoencoder

This project focuses on detecting fraudulent credit card transactions using both supervised and unsupervised deep learning methods. The objective is to compare how well an Artificial Neural Network (ANN) and an Autoencoder perform in identifying fraudulent activity.

---

## Project Overview

This project implements two approaches:

### 1. Artificial Neural Network (Supervised)

* Uses SMOTE to address class imbalance
* Trained to classify fraudulent vs. legitimate transactions
* Evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC

### 2. Autoencoder (Unsupervised Anomaly Detection)

* Trained only on normal (non-fraud) transactions
* Fraud is detected by calculating reconstruction error
* Evaluated using confusion matrix and ROC-AUC

---

## Dataset

This project uses the **Credit Card Fraud Detection dataset from Kaggle (creditcard.csv)**.
The dataset contains anonymized transaction features and a binary target label indicating whether a transaction is fraudulent.

---

## Preprocessing Steps

* Removed missing values
* Converted target column to integer
* Standardized all numerical features
* Applied SMOTE on the training data (ANN only)
* Performed stratified train/test split

---

## Model Architectures

### ANN Classifier

* Dense layer with 64 units (ReLU)
* Dropout layer with 0.4 rate
* Dense layer with 32 units (ReLU)
* Output layer with sigmoid activation
* Uses early stopping to prevent overfitting

### Autoencoder

* Encoder: 32 → 16 units
* Decoder: 32 → output dimension
* Optimized using mean squared error reconstruction loss

---

## Results Summary

### ANN Performance

* Demonstrates high accuracy and recall for detecting fraud
* Achieves a strong ROC-AUC score
* Performs effectively with the help of SMOTE for balance

### Autoencoder Performance

* Performs well without relying on labeled fraud data
* Achieves competitive ROC-AUC
* Lower recall compared to the ANN, which is expected for unsupervised methods

---

## Visual Outputs Included

The project includes:

* ANN training loss plot
* Confusion matrix for ANN
* ROC curve for Autoencoder
* Additional performance visualizations

---

## How to Run the Project

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the main script:

```bash
python credit_card_fraud_detection.py
```

Ensure that **creditcard.csv** is placed in the project directory before running.

---

## Conclusion

The ANN model provides superior performance in detecting fraudulent transactions due to the advantages of supervised learning combined with SMOTE.
The Autoencoder serves as a useful unsupervised alternative, particularly in scenarios where fraud labels are limited or unavailable.

