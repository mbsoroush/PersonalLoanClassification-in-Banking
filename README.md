# 🏦 Personal Loan Classification in Banking

A supervised machine learning project that predicts **which bank customers are most likely to accept a personal loan offer**. It is based on a case in which Thera Bank wants to convert its liability (deposit) customers into personal loan customers while keeping them as depositors. The previous campaign converted just over 9% of customers, and the marketing team wants better targeting to raise that rate on a smaller budget. The project cleans and explores the customer data, compares Logistic Regression, Naive Bayes, K-Nearest Neighbors, and Random Forest, then adds ensemble methods. The best model, a **Random Forest, reaches 99.4% accuracy**.

![Python](https://img.shields.io/badge/Language-Python%203.10-3776AB?style=flat-square)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-189AB4?style=flat-square)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-6A5ACD?style=flat-square)

## 📦 Dataset

The data is hosted on Kaggle and is not stored in this repository:

👉 [Personal Loan Dataset on Kaggle](https://www.kaggle.com/datasets/mbsoroush/personal-loan-dataset) (`Bank_Personal_Loan_Modelling.csv`)

**5,000 customers × 14 columns**, with no missing values and no duplicate rows.

**Target:** `Personal Loan`: did the customer accept the loan offered in the last campaign? (`0` = no, `1` = yes). Only **about 9.6%** of customers accepted (480 of 5,000), so the classes are imbalanced.

| Group | Features |
|-------|----------|
| **Demographics** | `Age`, `Experience` (years of professional experience), `Family` (family size), `Education` (1 undergraduate, 2 graduate, 3 advanced/professional) |
| **Finances** | `Income` (annual), `CCAvg` (average monthly credit card spending), `Mortgage` (value of house mortgage) |
| **Bank relationship** | `Securities Account`, `CD Account` (certificate of deposit), `Online` (internet banking), `CreditCard` (card issued by this bank) |
| **Dropped** | `ID`, `ZIP Code` |

## 🔄 Workflow

1. **Cleaning:** `Experience` contains negative values (minimum −3), so they are converted to absolute values. `ID` and `ZIP Code` are dropped, which leaves 11 features and the target.
2. **EDA:** distribution and box plots for `Age`, `Experience`, `Income`, `CCAvg`, and `Mortgage`; count plots of every categorical feature against loan acceptance; a correlation heatmap; and a pair plot.
3. **Modeling:** each algorithm gets a baseline, and where applicable a tuned version using `GridSearchCV` hyperparameters. Features are scaled (standardization or min-max, depending on the model).
4. **Evaluation:** accuracy, macro precision, recall, F1, confusion matrices, precision-recall and ROC curves, and 10-fold cross-validation.
5. **Ensembles:** Voting, Bagging, AdaBoost, Gradient Boosting, and XGBoost.
6. **Prediction:** a trained model scores a customer record.

## 🔍 What Drives Loan Acceptance?

Correlation with `Personal Loan`:

| Feature | Correlation |
|---------|:-----------:|
| **`Income`** | **0.50** |
| `CCAvg` | 0.37 |
| `CD Account` | 0.32 |
| `Education` | 0.14 |
| `Mortgage` | 0.14 |
| `Family` | 0.06 |
| `Age`, `Experience`, `Securities Account`, `Online`, `CreditCard` | about 0 |

Among the features themselves, `Age` and `Experience` are almost identical (0.99), `Income` and `CCAvg` move together (0.65), and `CD Account` is linked to `Securities Account` (0.32) and `CreditCard` (0.28).

## 📊 Results

| Model | Key settings | Test split | Accuracy | Macro F1 | 10-fold CV |
|-------|--------------|:----------:|:--------:|:--------:|:----------:|
| Logistic Regression | defaults | 20% | 0.952 | 0.844 | 0.948 |
| Improved Logistic Regression | L1 penalty, `C=10`, `liblinear` | 20% | 0.953 | 0.848 | 0.952 |
| Gaussian Naive Bayes | min-max scaling | 25% | 0.886 | 0.717 | 0.883 |
| Complement Naive Bayes | binned + one-hot features | 20% | 0.876 | 0.733 | 0.737 |
| K-Nearest Neighbors | `k=3` | 30% | 0.967 | 0.891 | 0.910 |
| Improved KNN | `k=4`, distance-weighted | 30% | 0.968 | 0.896 | 0.912 |
| **Random Forest** | defaults | 20% | **0.994** | **0.983** | **0.988** |
| Voting Classifier | soft voting over 7 models | 20% | 0.971 | 0.906 | 0.966 |
| Bagging | 700 bagged KNN (`k=3`) | 20% | 0.972 | 0.909 | 0.913 |
| AdaBoost | 200 estimators, `learning_rate=0.1` | 20% | 0.970 | 0.903 | 0.969 |
| Gradient Boosting | 500 estimators, `learning_rate=0.1` | 20% | 0.989 | 0.968 | 0.987 |
| XGBoost | 900 estimators, `learning_rate=0.1` | 20% | 0.987 | 0.962 | 0.986 |

Cross-validation setups vary slightly between rows (see the notebook).

### ⚖️ Why accuracy alone is misleading

Because about 90.4% of customers decline, a model that predicts "no" for everyone would already score about 90.4% accuracy. The two Naive Bayes models score *below* that baseline. A better test of usefulness is how many of the actual acceptors a model finds:

| Model | Precision (acceptors) | Recall (acceptors) | F1 (acceptors) |
|-------|:---------------------:|:------------------:|:--------------:|
| Logistic Regression | 0.83 | 0.62 | 0.71 |
| Gaussian Naive Bayes | 0.43 | 0.59 | 0.50 |
| K-Nearest Neighbors | 0.94 | 0.69 | 0.80 |
| Gradient Boosting | 0.97 | 0.92 | 0.94 |
| **Random Forest** | **0.98** | **0.96** | **0.97** |

### 🏆 Best model: Random Forest

The Random Forest correctly identifies 96% of the customers who go on to accept a loan, with 98% precision, while making almost no mistakes on the rest. Gradient Boosting and XGBoost are close behind, and the simpler models (Logistic Regression, KNN, Naive Bayes) miss many acceptors.

## 🔮 Predicting for a New Customer

The final section shows how to score a customer record with a trained model, using a sample from the dataset.

## 📁 Repository Structure

```
PersonalLoanClassification/
├── personal-loan-classification-in-banking.ipynb   # EDA, modeling, ensembles, evaluation
└── README.md
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone <this-repository-url>
   cd <repository-folder>
   ```
2. Install the dependencies (the notebook was run with Python 3.10):
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
   ```
   > The Bagging cell uses `base_estimator=`, which was **removed in scikit-learn 1.4**. Use `scikit-learn<1.4`, or rename it to `estimator=` on newer versions.
3. Download `Bank_Personal_Loan_Modelling.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mbsoroush/personal-loan-dataset) and place it in the project folder.
4. Open the notebook and update the path in the data-loading cell to where you saved the file. The default is a Kaggle path:
   ```python
   data = pd.read_csv('path/to/Bank_Personal_Loan_Modelling.csv')
   ```
5. Launch Jupyter and run the cells in order:
   ```bash
   jupyter notebook personal-loan-classification-in-banking.ipynb
   ```

> Newer seaborn versions deprecate `sns.distplot`. If you hit warnings or errors, replace it with `histplot`/`displot`.

## 🎯 Learning Objectives

- Clean and explore a banking dataset and spot which features relate to the target
- Work with an imbalanced classification problem and look beyond accuracy
- Compare linear, probabilistic, distance-based, and tree-based classifiers
- Tune hyperparameters and judge the effect with test-set and cross-validated scores
- Evaluate models with confusion matrices, precision-recall curves, and ROC curves
- Combine models with voting, bagging, and boosting
