# Logistic-Regression-in-classification
It is a logistic regression loan appliction verified with client age,salary,credit score,exp_level
Loan Default Prediction using Logistic Regression (with GridSearchCV)
## Project Overview

This project is an end-to-end machine learning classification system that predicts whether a customer will default on a loan based on their demographic and financial details.

The project demonstrates the complete ML workflow used in real-world and product-based companies:

Data preprocessing

Feature scaling

Model training

Hyperparameter tuning using GridSearchCV

Model evaluation

Prediction on new user input

## Problem Statement

Financial institutions face high risk when approving loans.
The goal of this project is to predict loan default risk using historical customer data.

Target Variable

defaulted

0 → No default

1 → Default

## Machine Learning Approach

Algorithm: Logistic Regression

Type: Binary Classification

Why Logistic Regression?

Simple and interpretable

Works well for binary outcomes

Widely used in finance and risk modeling

## Dataset Features

The dataset includes the following features:

Feature	Description
age	Customer age
salary	Annual salary
experience_years	Work experience
credit_score	Credit score
loan_amount	Loan amount
education_level_*	Education (One-Hot Encoded)
defaulted	Target variable
## Technologies Used

Python 

Pandas

NumPy

Scikit-learn

GridSearchCV

## Project Workflow

Load and inspect the dataset

Separate features and target

Split data into training and testing sets

Apply StandardScaler for feature scaling

Train Logistic Regression model

Tune hyperparameters using GridSearchCV

Evaluate model using:

Accuracy

Confusion Matrix

Classification Report

Predict loan default for new user input

## Hyperparameter Tuning (GridSearchCV)

To improve model performance, GridSearchCV is used with 5-fold cross-validation.

Tuned Parameters

C → Regularization strength

penalty → L2 regularization

solver → Optimization algorithm

This ensures better generalization and prevents overfitting.

## Model Evaluation Metrics

Accuracy Score

Confusion Matrix

Precision, Recall, F1-score

These metrics help in understanding both overall performance and class-wise behavior.

## Sample Prediction (User Input)

The model allows real-time prediction by taking user input such as:

Age

Salary

Experience

Credit score

Loan amount

Education level

The input is preprocessed using the same scaler used during training to avoid data leakage.

## Key Learnings

Importance of feature scaling in Logistic Regression

Proper use of GridSearchCV for hyperparameter tuning

Handling categorical variables using one-hot encoding

Avoiding common ML mistakes like feature mismatch and data leakage

## Future Improvements

Add ROC-AUC curve and threshold tuning

Save model using joblib

Deploy using Flask or FastAPI

Add automated input validation

## Author

Vikas Thakur
MCA Student |  Machine Learning Engineer


git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction
python main.py
