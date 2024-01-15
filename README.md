# Credit Card Fraud Detection Model

## Overview

This repository contains code for a credit card fraud detection model using an autoencoder-based approach. The model is built using TensorFlow and Keras, and logistic regression is used for the final classification.

## Contents

- `creditcard.csv`: Dataset containing credit card transactions.
- `credit_card_fraud_detection.ipynb`: Jupyter Notebook with the code for building and training the fraud detection model.
- `README.md`: Documentation file providing an overview of the project.

## Usage

1. **Dependencies**: Ensure you have the required dependencies installed. You can install them using the following:

    ```bash
    pip install pandas numpy matplotlib scipy tensorflow seaborn scikit-learn
    ```

2. **Dataset**: Make sure you have the `creditcard.csv` dataset in the same directory as the Jupyter Notebook.

3. **Run the Jupyter Notebook**: Open and run the `credit_card_fraud_detection.ipynb` notebook. This notebook contains all the code for loading the dataset, preprocessing, building the autoencoder model, training, and evaluating the fraud detection model.

4. **Review Results**: After running the notebook, review the classification report and accuracy score to assess the model's performance.

## Model Architecture

The fraud detection model is built using an autoencoder neural network. The encoder part reduces the dimensionality of the input features, and the decoder part reconstructs the input. The hidden representation obtained from the encoder is then used for training a logistic regression classifier.

## Dataset Information

The dataset contains credit card transactions with features like time, amount, and various anonymized numerical features. The target variable is 'Class,' where 1 indicates a fraudulent transaction and 0 indicates a normal transaction.

## Results

The logistic regression classifier achieves an accuracy of approximately 95.3% on the validation set, with high precision and recall for both normal and fraudulent transactions.

## References

- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Autoencoder Neural Network](https://en.wikipedia.org/wiki/Autoencoder)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Author

Raghavendran S


