# Loan Default Prediction Using Machine Learning

## Project Overview

This project was developed for the Data Science for Business Applications course at the Higher Colleges of Technology (HCT).

The project aims to predict potential loan default risks using machine learning techniques and demonstrate their application in financial risk assessment.

## Live Application

[Launch the Loan Default Prediction Application](https://loan-default-prediction-cds.streamlit.app/)

## Business Objectives

- Identify potential loan default risks.
- Support data-driven financial decision-making.
- Compare machine learning classification models.
- Develop an interactive prediction application.

## Dataset

- Source: Kaggle – Loan Default Dataset
- Total Records: 255,347
- Original Columns: 18
- Target Variable: Default
- Non-Default: 88.39%
- Default: 11.61%

[Dataset Source](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

## Project Methodology

1. Business Problem Formulation
2. Dataset Collection
3. Data Understanding and Cleaning
4. Exploratory Data Analysis and Feature Engineering
5. Model Selection and Development
6. Hyperparameter Tuning and Evaluation
7. Application Development and Deployment

## Feature Engineering

Three additional features were created:

- Loan-to-Income Ratio
- Monthly Income
- Estimated Monthly Principal

## Machine Learning Models

The following classification models were developed and compared:

- Logistic Regression
- Random Forest

RandomizedSearchCV and stratified three-fold cross-validation were used for hyperparameter optimization.

**Selected Final Model: Random Forest Classifier**

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Application Features

- Interactive borrower information form
- Financial and loan information inputs
- Real-time loan default predictions
- Estimated default risk score
- Financial summary
- Model performance information

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab
- GitHub
- Streamlit Community Cloud

## Repository Structure

| File | Description |
|---|---|
| Jupyter Notebook | Data analysis, model training, and evaluation |
| app.py | Streamlit application source code |
| loan_default_model.pkl | Trained Random Forest pipeline |
| model_info.json | Model information and evaluation metrics |
| app_settings.json | Application input configurations |
| requirements.txt | Required software dependencies |
| README.md | Project documentation |

## Run the Application Locally

Install the required dependencies:

`pip install -r requirements.txt`

Run the application:

`streamlit run app.py`

## Limitations and Disclaimer

This project was developed for educational purposes. The model's estimated risk scores are not calibrated probabilities and should not be used for actual loan approval or rejection decisions.

Further validation, fairness assessments, and probability calibration are required before real-world financial implementation.

## Institution

Higher Colleges of Technology – Sharjah

Computer Information Science – Data Science

Data Science for Business Applications

2026
