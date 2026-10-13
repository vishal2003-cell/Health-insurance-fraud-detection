# HealthShield — Health Insurance Fraud Detection

A Machine Learning web application that predicts the likelihood of health insurance fraud based on claim-related information.

## Live Demo

[Open HealthShield Web App](https://health-insurance-fraud-detection-sjq7ktfdymem6dz938zmn6.streamlit.app/)

## Project Overview

HealthShield is an educational machine learning project developed to demonstrate how classification algorithms can help identify potentially fraudulent health insurance claims.

The application uses a Logistic Regression model and a Streamlit interface to display a prediction based on the information entered by the user.

## Features

- Interactive web application built with Streamlit.
- Machine learning predictions using Logistic Regression.
- Input fields for patient and insurance claim details.
- Displays predicted claim status and estimated fraud probability.
- Uses saved model, scaler, and feature-column files.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Input Information

The application accepts information such as:

- Age and gender
- Hospital type
- Emergency claim status
- Claim amount and treatment cost
- Previous claims count
- Policy holding years
- Days hospitalized

## How It Works

1. The user enters the insurance claim details.
2. The application prepares the input features.
3. The saved preprocessing scaler transforms the relevant numerical features.
4. The trained Logistic Regression model generates a prediction.
5. The application displays the predicted status and estimated fraud probability.

## Run Locally

Clone this repository:

```bash
git clone https://github.com/vishal2003-cell/Health-insurance-fraud-detection.git
cd Health-insurance-fraud-detection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Files

- `app.py` — Streamlit application.
- `logisticRegression.pkl` — Saved machine learning model.
- `scaler.pkl` — Saved feature scaler.
- `columns.pkl` — Saved model feature-column information.
- `requirements.txt` — Python dependencies.

## Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions are not a verified determination of fraud and should not be used to make real insurance, financial, or legal decisions.

## Author

**Vishal Kumar**

GitHub: [vishal2003-cell](https://github.com/vishal2003-cell)
