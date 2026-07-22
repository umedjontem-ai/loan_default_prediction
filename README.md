# loan_default_prediction models | Data science
Dataset source: 
NIKHIL(2023). https://www.kaggle.com/datasets/nikhil1e9/loan-default

# Loan Default Prediction using Machine Learning

## Project Overview
This project develops a machine learning pipeline to predict whether a customer is likely to default on a loan. The objective is to help financial institutions identify high-risk borrowers and support data-driven lending decisions.
The project covers the complete machine learning workflow, including data preprocessing, feature engineering, model development, evaluation, threshold optimization, and model persistence.

---

## Business Problem
Loan defaults can result in significant financial losses for banks and lending institutions. The goal of this project is to build a binary classification model capable of estimating the probability that a customer will default on a loan.
Since identifying risky borrowers is often more important than maximizing overall accuracy, special attention was given to handling class imbalance and optimizing the classification threshold.

---

## Dataset
* **Records:** 255,347 loan applications
* **Target Variable:** `Default`

  * `0` = No Default
  * `1` = Default
    
The dataset contains customer demographic and financial information, including:

* Age
* Income
* Loan Amount
* Credit Score
* Employment Duration
* Interest Rate
* Debt-to-Income Ratio (DTI)
* Education
* Employment Type
* Marital Status
* Mortgage Status
* Dependents
* Loan Purpose
* Co-signer Information

---

## Exploratory Data Analysis (EDA)
The dataset was explored to understand its structure and quality.
Performed analyses included:

* Dataset overview
* Missing value analysis
* Duplicate detection
* Data type inspection
* Statistical summaries
* Target class distribution
* Feature visualization
* Correlation analysis

The dataset contained no missing values and was suitable for model development after preprocessing.

---

## Feature Engineering
Several new features were created to improve predictive performance:

* Loan-to-Income Ratio
* Credit Score Categories

Categorical variables were transformed using One-Hot Encoding where appropriate.

---

## Data Preprocessing
The preprocessing pipeline included:

* Removing unnecessary features
* Feature engineering
* One-Hot Encoding
* Feature Scaling using StandardScaler (for Logistic Regression)
* Train/Test Split
* Handling class imbalance using class weights and XGBoost's `scale_pos_weight`

---

## Machine Learning Models
The following classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Each model was evaluated using identical train/test splits for fair comparison.

---

## Evaluation Metrics
The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Because the dataset is imbalanced, Recall and ROC-AUC were considered more informative than Accuracy.

---

## Model Comparison

| Model               | ROC-AUC   |
| ------------------- | --------- |
| Logistic Regression | **0.762** |
| XGBoost             | 0.756     |
| Random Forest       | 0.746     |
| Decision Tree       | 0.729     |

Logistic Regression achieved the best overall performance while maintaining strong recall for the minority class.

---

## Threshold Optimization
Instead of relying solely on the default classification threshold of **0.50**, multiple thresholds were evaluated.
Lowering the threshold to **0.40** increased Recall from approximately **70%** to **82%**, improving the model's ability to detect customers likely to default.
This trade-off is particularly valuable in credit risk assessment, where missing a high-risk borrower is often more costly than incorrectly flagging a low-risk applicant.

---

## Feature Importance
Feature importance analysis identified the most influential predictors, including:

* Age
* Loan-to-Income Ratio
* Interest Rate
* Income
* Months Employed

These features contributed most to predicting loan default risk.

---

## Model Persistence
The final trained model was saved using **Joblib**, along with:

* Trained Logistic Regression model
* StandardScaler
* Feature list

This enables the model to be reused without retraining.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Joblib
* Jupyter Notebook

---

## Key Takeaways

* Built a complete end-to-end machine learning pipeline.
* Applied feature engineering to improve predictive performance.
* Compared multiple machine learning algorithms.
* Addressed class imbalance using appropriate techniques.
* Optimized the classification threshold for a business-oriented objective.
* Evaluated models using industry-standard metrics.
* Saved the final model for future deployment.
