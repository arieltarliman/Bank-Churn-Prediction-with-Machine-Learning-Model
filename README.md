# Bank Customer Churn Prediction  
*Machine Learning Project – Semester 3, 2024*  

---

## Overview  
This project applies machine learning to predict **bank customer churn** using a real dataset from Kaggle:  
[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data).  

Customer churn is a critical issue for banks as it impacts revenue, customer lifetime value, and long-term growth. The aim is to analyze customer demographics, financial data, and account activity to build a model that identifies customers at risk of leaving, enabling proactive retention strategies.  

---

## Key Insights  
- The dataset is **imbalanced**: more customers stay than churn.  
- **Key churn drivers**:  
  - Older customers with fewer bank products.  
  - Customers who are inactive or lack a credit card.  
  - Geography and gender differences influence churn rates.  
- A **Random Forest Classifier** performed well and was further optimized with **GridSearchCV** and **threshold adjustment**.  
- ROC AUC of **0.81** shows the model can distinguish churners from non-churners effectively.  

---

## Files in this Repository  
- **[`data/2702234636-ArieldhiptaTarliman_GSLC.ipynb`](./src/2702234636-ArieldhiptaTarliman_GSLC.ipynb)**  
  Jupyter Notebook with preprocessing, EDA, modeling, and evaluation.  

- **[`src/Bank Customer Churn Prediction.csv`](./data/Bank%20Customer%20Churn%20Prediction.csv)**  
  Dataset used in this project, downloaded from Kaggle.  

---

## Methods  
1. **Exploratory Data Analysis (EDA)**
   - Check on Missing Values, Duplicated Values, Unique Values, Outliers of data
   - Distribution of churn vs non-churn across features.
   - Distribution of numerical columns.
   - Correlation analysis between  variables and target (churn).
     
3. **Data Preprocessing**  
   - One Hot Encoding categorical variables (Gender, Geography).  
   - Robust Scaling for numerical variables.
   - Ordinal encoding for ordinal features.

4. **Modeling & Evaluation**  
   - Trained classification models with RandomForestClassifier.
   - Fine tuned with GridSearchCV.
   - Evaluated using Confusion Matrix, Accuracy, Precision, Recall, and F1-score, and  ROC Curve.
---

## Results  

### Baseline Random Forest  
- **Test Accuracy**: 0.85  
- **Classification Report**:  
- Precision (Class 0: stayed): 0.88 | Recall: 0.95 | F1: 0.91  
- Precision (Class 1: churned): 0.68 | Recall: 0.45 | F1: 0.54  
- Weighted Avg F1: 0.84  

---

### After GridSearchCV (Best Parameters)  
- **Test Accuracy**: 0.837  
- **Classification Report**:  
- Precision (Stayed): 0.88 | Recall: 0.93 | F1: 0.90  
- Precision (Churned): 0.61 | Recall: 0.47 | F1: 0.53  
- Weighted Avg F1: 0.83  

---

### With Optimal Threshold (0.24, maximizing recall for churn)  
- **Test Accuracy**: 0.77  
- **Classification Report**:  
- Precision (Stayed): 0.92 | Recall: 0.79 | F1: 0.85  
- Precision (Churned): 0.45 | Recall: 0.71 | F1: 0.55  
- Weighted Avg F1: 0.79  

---

### ROC Curve  
- **AUC = 0.81**  
- Indicates the model performs **fairly well** at distinguishing churners vs non-churners.  
- AUC closer to 1.0 would be perfect; 0.5 would indicate random guessing.  

---

## How to Reproduce  
### Requirements  
- Python (≥ 3.8)  
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`  
