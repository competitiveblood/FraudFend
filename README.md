# FraudFend
 
## Mobile Fraud Detection System

This project implements a mobile fraud detection system using machine learning and deep learning techniques. The system identifies fraudulent transactions in financial datasets with high accuracy, while addressing class imbalance issues.

## Features
- Preprocessing pipeline to clean and prepare transaction data.
- Combines numerical feature analysis and customer ID embeddings for fraud prediction.
- Utilizes oversampling and class weights to handle class imbalance.
- Includes metrics like Accuracy, AUC, Confusion Matrix, and Classification Report for performance evaluation.

---

## Dataset Information
The dataset contains transaction details such as:
- **Type**: Type of transaction (e.g., CASH_OUT, TRANSFER).
- **Amount**: Transaction amount.
- **nameOrig**: Customer ID of the origin.
- **nameDest**: Customer ID of the destination.
- **isFraud**: Target column (1 = Fraud, 0 = Not Fraud).

---

## Data Preprocessing
1. **Dropped Unnecessary Columns**:
   - Removed `step`, `nameOrig`, and `isFlaggedFraud` as they do not contribute to the analysis.
2. **One-Hot Encoding**:
   - Converted the categorical `type` column into binary indicator columns for interpretability.
3. **Train-Test Split**:
   - Split the dataset into 70% training and 30% testing data.
4. **Tokenized Customer IDs**:
   - Tokenized `nameDest` and padded for uniform input size.
5. **Standard Scaling**:
   - Normalized numerical features using `StandardScaler`.

---

## Handling Class Imbalance
Fraudulent transactions were underrepresented in the dataset. To address this:
- **Oversampling**: Duplicated fraudulent samples to balance the dataset.
- Result: Equal representation of fraud (`isFraud=1`) and non-fraud (`isFraud=0`) classes.

---

## Model Architecture
1. **Inputs**:
   - **Numerical Features**: Transaction-related features such as amount and balance.
   - **Customer IDs (nameDest)**: Embedded using an embedding layer.
2. **Layers**:
   - Dense layers with ReLU activation for numerical features.
   - Embedding layer for tokenized customer IDs.
   - Concatenated features from both inputs.
3. **Output**:
   - Sigmoid activation for binary classification (fraud/non-fraud).
4. **Compilation**:
   - Optimizer: `Adam`
   - Loss Function: `Binary Cross-Entropy`
   - Metrics: `Accuracy` and `AUC`
5. **Training**:
   - Used class weights to address residual imbalance.
   - Early stopping to prevent overfitting.

---

## Evaluation Metrics
- **Accuracy**: Percentage of correctly classified transactions.
- **AUC (Area Under the Curve)**: Ability of the model to distinguish between fraud and non-fraud cases.
- **Confusion Matrix**:
  - Visualizes True Positives, True Negatives, False Positives, and False Negatives.
  - Provides insights into model errors.
- **Classification Report**:
  - Precision, Recall, and F1-Score for both classes.

---

## Visualization
- **Heatmap of Confusion Matrix**:
  - Identifies areas for improvement in the model's predictions.

---

## Results
The fraud detection system achieved:
- **High Accuracy** and interpretability.
- **Effective handling of class imbalance** using oversampling and class weights.
- **Actionable insights** through evaluation metrics and visualization.

---

## Saved Model
The trained model is saved as:
- **Filename**: `my_model.keras`
- Format: HDF5

---
