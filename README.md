# 🏠 Rent Interest Classifier

A machine learning pipeline for predicting renter interest levels (low, medium, high) in real estate listings using structured data and sentiment analysis of listing descriptions. This project explores feature engineering, class balancing, and model performance across traditional and deep learning approaches.

## 📊 Overview

We used the Kaggle dataset ["Two Sigma Connect: Rental Listing Inquiries"](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) and trained multiple models to classify interest levels:

- Decision Tree
- Random Forest
- Deep Neural Network (LSTM + FCNN)
- DistilBERT for sentiment extraction

Key techniques include:
- Sentiment analysis using RNN and DistilBERT
- Multi-input deep learning architecture (text + structured features)
- Data balancing via oversampling
- Evaluation with cross-validation and classification metrics

## 🧠 Models Used

| Model           | Accuracy (Balanced + Sentiment) |
|----------------|------------------------------|
| Decision Tree  | 43.55%                       |
| Random Forest  | **95.31%**                   |
| DNN (Untuned)  | 70.92%                       |
| DNN (Tuned)    | 69.44%                       |

## ⚙️ Technologies

- Python
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Optuna
- Pandas, NumPy, Matplotlib, Seaborn

Sentiment was derived from the description column using both RNN and DistilBERT. Final predictions were made using Random Forest and deep learning models with and without the sentiment feature.


