# CA03_DecisionTrees

# Decision Tree Algorithm for Income Classification

This repository contains a complete end-to-end project on building and tuning a Decision Tree classifier to predict whether an individual's income is greater than \$50K or not. The project includes data quality analysis, data cleansing, feature transformation, model training, hyperparameter tuning, model evaluation, visualization of the decision tree, and making predictions on new data.

## Overview

The dataset is obtained from the U.S. Census Bureau and includes various demographic features (already discretized into bins) and an income label (<=50K or >50K). The primary goals of this project are to:
- Conduct a detailed Data Quality Analysis (DQA) on the dataset.
- Clean, preprocess, and encode the data.
- Build a Decision Tree classifier using scikit-learn.
- Systematically tune key hyperparameters (split criterion, minimum samples per leaf, maximum features, and maximum depth) using iterative experiments.
- Visualize model performance and the structure of the best-performing tree.
- Make a prediction for a new individual using the final model, including a confidence score.

## Dataset

- **Source:** U.S. Census Bureau  
- **Description:** The dataset contains demographic information including:
  - `hours_per_week_bin`
  - `occupation_bin`
  - `msr_bin` (Marriage Status & Relationships)
  - `capital_gl_bin`
  - `race_sex_bin`
  - `education_num_bin`
  - `education_bin`
  - `age_bin`
  - `workclass_bin`
  - `flag` (indicating if the row is for training, testing, or unknown)
- **Target Variable:** `y` (income category: 0 for <=50K, 1 for >50K)

## Project Structure

- **notebook.ipynb:** Jupyter Notebook containing all the code for data exploration, cleaning, model training, hyperparameter tuning, and evaluation.
- **README.md:** This file, providing an overview and instructions.
- **requirements.txt:** List of Python packages required for the project (e.g., scikit-learn, pandas, matplotlib, seaborn, etc.).

## Requirements

Ensure you have the following packages installed:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.tree import plot_tree


In order to run:
-git clone https://github.com/yourusername/your-repo-name.gitcd your-repo-name
-Open the notebook file
-Run cells sequentially (they are in sections as follows:)
Data Quality Analysis (DQA)
Data Cleansing and Transformation (including encoding)
Model Building using Decision Tree Classifier
Hyperparameter Tuning (four runs varying different hyperparameters)
Model Evaluation (accuracy, precision, recall, F1 score, ROC AUC)
Final Model Visualization
Making Predictions on a New Data Record


Here is some more details about this project:
Data Quality Analysis:
The notebook explores the dataset for missing values, outliers, and skewness, and creates a detailed Data Quality Report.

Data Cleaning and Encoding:
Categorical features are preprocessed (using regex to remove unwanted prefixes) and encoded using LabelEncoder. The same preprocessing is applied to new records for prediction consistency.

Model Training and Hyperparameter Tuning:
The Decision Tree model is trained using the training set. Hyperparameters are tuned in four runs:

Run 1: Varying split criterion (gini vs. entropy)
Run 2: Varying minimum samples per leaf
Run 3: Varying maximum features
Run 4: Varying maximum depth
Each run is evaluated using accuracy along with precision, recall, and F1 score. Line graphs visualize the performance changes.
Final Model and Visualization:
The best-performing tree is built using the optimal hyperparameters, and its structure is visualized for interpretability.

Prediction on New Data:
A new record is created (using the same feature structure and encoding as the training data), and the final model predicts the income category with an associated confidence (probability) score.

Conclusion
This project demonstrates a systematic approach to building, tuning, and evaluating a Decision Tree classifier for income classification. The detailed hyperparameter tuning process ensures that the final model is both robust and interpretable, while the prediction on a new individual illustrates the practical application of the model.

