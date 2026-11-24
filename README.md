# Credit Card Fraud Detection

A machine learning project for identifying fraudulent credit card transactions using classification algorithms, class imbalance handling, model evaluation, and hyperparameter optimization. The project explores multiple approaches to improve fraud detection on a highly imbalanced dataset, including SMOTE, Balanced Random Forest, and GPU-accelerated XGBoost.

## Overview

Credit card fraud detection is a binary classification problem in which the objective is to distinguish legitimate transactions from fraudulent ones.

One of the biggest challenges is **class imbalance**: fraudulent transactions represent only a very small fraction of all transactions. Consequently, a model can achieve extremely high accuracy while failing to identify actual fraud cases.

This project investigates different machine learning models and techniques to address this challenge, with particular emphasis on fraud detection performance rather than accuracy alone.

## Key Features

- Exploratory Data Analysis (EDA) and data visualization.
- Data preprocessing and preparation for classification.
- Comparison of multiple machine learning algorithms.
- Handling class imbalance using SMOTE.
- Balanced Random Forest classification.
- Hyperparameter optimization using GridSearchCV.
- Probability-based evaluation using ROC-AUC.
- GPU acceleration for XGBoost using NVIDIA CUDA.
- Model comparison using multiple classification metrics.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset, containing anonymized transaction features and a binary fraud label.

The dataset includes:

| Feature | Description |
|---|---|
| `Time` | Time elapsed between the transaction and the first transaction in the dataset |
| `V1`–`V28` | Anonymized features obtained through PCA transformation |
| `Amount` | Transaction amount |
| `Class` | Target variable: `0` = Legitimate, `1` = Fraudulent |

The dataset is highly imbalanced, making conventional accuracy an insufficient measure of model performance.

Dataset location:

```text
dataset/creditcard.csv
```

## Machine Learning Workflow

The project follows an experimental workflow:

1. **Exploratory Data Analysis:** Examine the dataset, feature distributions, class distribution, and transaction characteristics.
2. **Preprocessing:** Prepare features and target variables, followed by training and testing data separation.
3. **Baseline Modeling:** Train initial classification models on the original imbalanced dataset.
4. **Class Imbalance Handling:** Apply SMOTE to generate synthetic minority-class samples and investigate Balanced Random Forest.
5. **Model Optimization:** Perform hyperparameter tuning using GridSearchCV.
6. **Evaluation:** Compare models using classification metrics and probability-based evaluation.

## Models Implemented

The project explores the following algorithms:

| Model | Library | Training |
|---|---|---|
| Logistic Regression | Scikit-learn | CPU |
| Decision Tree | Scikit-learn | CPU |
| Random Forest | Scikit-learn | CPU |
| XGBoost | XGBoost | NVIDIA GPU (CUDA) |
| Support Vector Machine (SVM) | Scikit-learn | CPU |
| Balanced Random Forest | Imbalanced-learn | CPU |
| K-Nearest Neighbors (KNN) | Scikit-learn | CPU |
| SGD Classifier | Scikit-learn | CPU |

Additional experiments include model calibration and soft voting.

## Handling Class Imbalance

Two approaches are explored:

### 1. SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates synthetic examples of the minority class to address the imbalance between fraudulent and legitimate transactions.

The resampled training data is used to train classification models and investigate whether fraud detection performance improves.

### 2. Balanced Random Forest

Balanced Random Forest uses balanced samples when training individual decision trees, helping the classifier learn from the minority class.

These approaches are compared against baseline models trained on the original data.

## Hyperparameter Optimization

GridSearchCV is used to investigate different hyperparameter configurations and select models based on ROC-AUC.

Experiments include:

- Logistic Regression hyperparameter tuning.
- SVM hyperparameter tuning with probability calibration.
- SGD Classifier hyperparameter tuning.
- Additional model comparison and voting experiments.

## GPU Acceleration

The project supports GPU-accelerated XGBoost training using an NVIDIA CUDA-compatible GPU.

The XGBoost GPU configuration uses:

```python
XGBClassifier(
    tree_method="hist",
    device="cuda",
    eval_metric="auc",
    random_state=42
)
```

**Important:** GPU acceleration is specific to XGBoost in this project. Scikit-learn models, SMOTE, preprocessing, and other CPU-based operations continue to run on the CPU.

The GPU configuration requires a compatible NVIDIA GPU, driver, and XGBoost installation.

## Evaluation Metrics

Model performance is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correctly classified transactions |
| Precision | Proportion of predicted fraud cases that are actually fraudulent |
| Recall | Proportion of actual fraud cases successfully detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Ability of the model to distinguish between legitimate and fraudulent transactions |

For fraud detection, recall and precision are particularly important because missing fraudulent transactions and incorrectly flagging legitimate transactions have different consequences.

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── CreditCardFraudDetection.ipynb
│
├── .gitattributes
├── .gitignore
└── README.md
```

- `dataset/creditcard.csv`: Transaction dataset.
- `CreditCardFraudDetection.ipynb`: Main notebook containing preprocessing, experiments, model training, tuning, and evaluation.
- `.gitattributes`: Git LFS configuration for the dataset.
- `.gitignore`: Excludes virtual environments, caches, and unnecessary local files.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer (use a version compatible with your installed ML packages).
- Git.
- Git LFS for downloading the dataset.
- Jupyter Notebook or Visual Studio Code.

### 1. Clone the repository

```bash
git clone https://github.com/UTKARSH2115/credit-fraud-detection.git

cd credit-fraud-detection
```

### 2. Download the dataset using Git LFS

```bash
git lfs install
git lfs pull
```

### 3. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm jupyter
```

### 5. Launch the notebook

```bash
jupyter notebook CreditCardFraudDetection.ipynb
```

Alternatively, open the notebook directly in Visual Studio Code.

## GPU Setup (Optional)

For NVIDIA GPU acceleration, use a compatible NVIDIA driver and an XGBoost installation with CUDA support.

The notebook uses:

```python
tree_method="hist"
device="cuda"
```

Verify GPU availability during training using:

```bash
nvidia-smi
```

GPU acceleration is optional. The remaining CPU-based experiments can run without an NVIDIA GPU.

## Future Improvements

- Use an imbalanced-learn Pipeline to prevent data leakage during cross-validation with SMOTE.
- Explore precision-recall curves and Average Precision (PR-AUC).
- Optimize classification thresholds according to fraud detection requirements.
- Experiment with additional ensemble and anomaly detection techniques.
- Evaluate models using stratified cross-validation.
- Investigate feature importance and model explainability using SHAP.
- Compare training time and resource utilization across CPU and GPU implementations.

## Technologies Used

- **Language:** Python
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Learning:** Imbalanced-learn (SMOTE, Balanced Random Forest)
- **Gradient Boosting:** XGBoost, LightGBM
- **GPU Computing:** NVIDIA CUDA
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git, GitHub, Git LFS

---

**Project:** Credit Card Fraud Detection  
**Repository:** [UTKARSH2115/credit-fraud-detection](https://github.com/UTKARSH2115/credit-fraud-detection)