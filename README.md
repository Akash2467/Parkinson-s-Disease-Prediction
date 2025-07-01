# Parkinson’s Disease Prediction using Machine Learning and Deep Learning

This project aims to predict Parkinson’s disease using a combination of machine learning and neural network models. It includes feature selection, model implementation, performance evaluation, dimensionality reduction using t-SNE, and neural network experimentation.

---

## Dataset

- **Name**: Parkinson’s Disease Classification  
- **Source**: UCI Machine Learning Repository – Dataset ID 174  
- **Description**: Biomedical voice measurements from individuals with and without Parkinson’s disease.  
- **Target**: `status` — 1 (Parkinson’s positive), 0 (healthy)

---

## Workflow Overview

### Feature Selection

- **Method**: Sequential Forward Selection (SFS)  
- **Base Estimator**: Logistic Regression  
- Reduced dimensionality while improving model efficiency.

### Model Implementation

The following models were implemented and trained on the selected features:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- Naïve Bayes  
- Generalized Linear Model (GLM)

### Evaluation Metrics

Each model was evaluated using:

- Accuracy Score  
- Classification Report (Precision, Recall, F1-Score)  
- Confusion Matrix (visualized via heatmap)  
- ROC Curve and AUC Score

---

## Dimensionality Reduction using t-SNE

- **Technique**: t-Distributed Stochastic Neighbor Embedding (t-SNE)  
- **Purpose**: Non-linear dimensionality reduction and visualization  
- Models were retrained on the reduced feature space for comparison.

### Model Performance Comparison

| Model                     | Accuracy Before t-SNE | Accuracy After t-SNE |
|---------------------------|------------------------|-----------------------|
| Logistic Regression       | 85%                    | 74%                   |
| Decision Tree             | 90%                    | 74%                   |
| Random Forest             | 90%                    | 85%                   |
| Support Vector Classifier | 85%                    | 90%                   |
| GLM                       | 87%                    | 81%                   |
| Gradient Boosting         | 90%                    | 90%                   |
| Naïve Bayes               | 79%                    | 79%                   |

### Observations

- Support Vector Classifier and Random Forest improved after t-SNE, indicating better feature representation.
- Logistic Regression and Decision Tree saw a drop in accuracy.
- Gradient Boosting and Naïve Bayes remained stable.
- t-SNE helped in visualizing complex relationships in high-dimensional data.

---

## Neural Network Implementation

Two neural network architectures were implemented:

### scikit-learn MLPClassifier

- Feedforward neural network  
- Accuracy: 95%

### TensorFlow ANN

- Optimized with Adam optimizer and ReLU activation  
- Accuracy: 92%

---

## Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- TensorFlow  
- Matplotlib, Seaborn  
- MLxtend  
- ucimlrepo

---
