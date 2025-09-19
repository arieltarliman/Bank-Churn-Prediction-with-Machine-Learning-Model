# Bank Customer Churn Prediction  
*Machine Learning Project – Semester 2, 2024*  

![Notebook Preview](./data/2702234636-ArieldhiptaTarliman_GSLC.ipynb)

---

## Overview  
This project focuses on predicting **bank customer churn** using supervised machine learning.  
The dataset is sourced from Kaggle:  
[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)  

Customer churn is a key challenge for banks as it directly impacts revenue and growth. The goal of this project is to analyze customer demographics and financial attributes to identify which customers are most likely to leave, and build a predictive model that helps in designing retention strategies.  

---

## Key Insights  
- The dataset is **imbalanced**: more customers stay than churn.  
- **Age, number of products, and account balance** are strong predictors of churn.  
- Customers who are **inactive** or do not hold a credit card are more likely to churn.  
- **Geography and gender** also influence churn behavior.  
- Predictive models achieve good performance, making it possible to target at-risk customers.  

---

## Files in this Repository  
- **[`data/2702234636-ArieldhiptaTarliman_GSLC.ipynb`](./data/2702234636-ArieldhiptaTarliman_GSLC.ipynb)**  
  Jupyter Notebook with preprocessing, EDA, modeling, and evaluation.  

- **[`src/Bank Customer Churn Prediction.csv`](./src/Bank%20Customer%20Churn%20Prediction.csv)**  
  Dataset used in this project, downloaded from Kaggle.  

---

## Methods  
1. **Data Preprocessing**  
   - Encoding categorical variables (Gender, Geography).  
   - Checking for missing values and data imbalances.  
   - Feature scaling and normalization where required.  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of churn vs non-churn across features.  
   - Correlation analysis between demographic/financial variables and churn.  

3. **Modeling & Evaluation**  
   - Trained classification models such as Logistic Regression, Decision Tree, and Random Forest.  
   - Evaluated using Accuracy, Precision, Recall, and F1-score.  
   - Compared results to determine the best performing model.  

---

## Results  
- **Correlation**: Older customers with fewer bank products are more likely to churn.  
- **Geography Impact**: Certain regions have higher churn rates than others.  
- **Engagement Factor**: Active members and credit card holders churn less often.  
- **Model Performance**: The best model achieved strong predictive results, showing practical value for churn prediction.  

---

## How to Reproduce  
### Requirements  
- Python (≥ 3.8)  
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`  

### Steps  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
