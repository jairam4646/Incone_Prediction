# Census Income Prediction Using CatBoost

## Project Overview

This project develops a machine learning pipeline to predict whether an individual's annual income exceeds $50,000 using demographic, educational, occupational, and financial attributes from the UCI Adult (Census Income) dataset.

The project follows a complete end-to-end workflow including data validation, exploratory data analysis, preprocessing, model training, hyperparameter tuning, threshold optimization, error analysis, and responsible machine learning evaluation.

---

## Problem Statement

Given demographic and employment-related information, predict whether an individual's income belongs to one of the following classes:

- Income > $50K
- Income ≤ $50K

This is a binary classification problem.

---

## Dataset

**Source:** UCI Adult (Census Income) Dataset

### Dataset Statistics

| Attribute | Value |
|------------|------------|
| Total Records | 48,842 |
| Training Records | 32,561 |
| Testing Records | 16,281 |
| Features | 14 |
| Target Variable | Income |

### Feature Types

#### Numerical Features

- age
- fnlwgt
- education_num
- capital_gain
- capital_loss
- hours_per_week

#### Categorical Features

- workclass
- education
- marital_status
- occupation
- relationship
- race
- sex
- native_country

#### Target Variable

- income (>50K / ≤50K)

---

## Project Structure

```text
.
├── 01_data_validation.ipynb
├── 02_eda.ipynb
├── 03_preprocess_policy.ipynb
├── 04_train_catboost.ipynb
├── 05_tuning.ipynb
├── 06_evaluation_thresholding.ipynb
├── 07_responsible_ml_checks.ipynb
├── inference.py
├── catboost_income_model.cbm
├── report.pdf
└── README.md
```

---

## Workflow

### 1. Data Validation

- Dataset loading and inspection
- Missing value identification
- Duplicate detection
- Data type verification
- Cardinality analysis

### 2. Exploratory Data Analysis (EDA)

- Univariate analysis
- Bivariate analysis with income
- Numerical feature distributions
- Categorical feature analysis
- Business insights generation

### 3. Preprocessing

- Missing value handling
- Duplicate removal
- Target encoding
- Categorical feature strategy
- Reproducible train-validation-test workflow

### 4. Model Development

- CatBoostClassifier
- Native categorical feature handling
- Baseline model training

### 5. Hyperparameter Tuning

- Randomized Search
- Validation AUC optimization
- Early stopping

### 6. Threshold Optimization

- Precision-Recall tradeoff analysis
- Threshold selection
- Final evaluation

### 7. Responsible ML Checks

- Group-wise evaluation by Sex
- Group-wise evaluation by Race
- Fairness assessment

---

## Model Configuration

### CatBoost

```python
loss_function="Logloss"
eval_metric="AUC"
random_seed=42
```

### Best Hyperparameters

| Parameter | Value |
|------------|------------|
| iterations | 1000 |
| learning_rate | 0.05 |
| depth | 6 |
| l2_leaf_reg | 5 |
| bagging_temperature | 0 |
| random_strength | 1 |
| one_hot_max_size | 2 |

---

## Final Model Performance

### Threshold = 0.40

| Metric | Value |
|------------|------------|
| ROC-AUC | 0.9275 |
| Accuracy | 86.66% |
| Precision | 70.86% |
| Recall | 73.92% |
| F1-Score | 72.36% |

---

## Key Findings

- Capital Gain is the strongest predictor of income.
- Relationship status strongly influences income classification.
- Education level is positively associated with higher income.
- Threshold optimization improved Recall and F1-Score.
- CatBoost effectively handled categorical variables without one-hot encoding.

---

## Responsible AI Considerations

The Adult dataset contains sensitive demographic attributes such as sex and race.

To promote responsible evaluation:

- Group-wise performance was analyzed across sex and race.
- Recall, AUC, FPR, and FNR were compared.
- Performance differences were documented and discussed.

This project is intended for educational purposes only and should not be used for real-world hiring, lending, admissions, or other high-impact decision-making scenarios.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- Google Colab
- Kaggle

---

## Author

**Jai Rama Srinivas**

Capstone Project: Census Income Prediction with CatBoost