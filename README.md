# Customer Churn Prediction Engine

Binary classification pipeline to predict customer churn probability using Logistic Regression with probability calibration.

## Pipeline Overview
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical
- **Class imbalance**: SMOTE oversampling on training set
- **Model selection**: 5-fold GridSearchCV over 15 combinations (Elastic Net penalty, SAGA solver, C ∈ {0.001–10}, l1_ratio ∈ {0.1, 0.5, 0.9})
- **Calibration**: CalibratedClassifierCV with Platt (sigmoid) scaling
- **Evaluation**: Train/Val/CV log loss tracked in parallel for overfitting detection

## Results
| Set | Log Loss |
|-----|----------|
| Training | tracked |
| Validation | tracked |
| CV Mean | tracked |

## Tech Stack
Python · Scikit-Learn · imbalanced-learn · Pandas · NumPy

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/Customer_Churn_Prediction_Using_Logistic_Regression.ipynb
```

## Key Design Decisions
- Elastic Net regularization combines L1 sparsity and L2 stability
- Platt scaling ensures predicted probabilities are well-calibrated (important for log loss evaluation)
- All transforms fitted on training set only — no leakage into validation
