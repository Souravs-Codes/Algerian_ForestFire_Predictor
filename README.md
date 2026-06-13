# 🔥 Algerian Forest Fire Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![AWS](https://img.shields.io/badge/AWS-Elastic_Beanstalk-yellow)

An end-to-end Machine Learning project that predicts the **Fire Weather Index (FWI)** using weather and environmental data from the Algerian Forest Fires dataset.

The project covers the complete ML workflow from data preprocessing and model training to cloud deployment on AWS using Flask.

---

## 🌐 Live Demo

## 📱 Mobile Access Note

If the application does not load on mobile devices or displays a connection error, make sure you are using the **HTTP** version of the URL instead of **HTTPS**.

✅ Working URL:

```text
http://algerianforestfirepredictor-env.eba-wa2ustx4.ap-south-1.elasticbeanstalk.com
```

❌ Do not use:

```text
https://algerianforestfirepredictor-env.eba-wa2ustx4.ap-south-1.elasticbeanstalk.com
```

The application is currently deployed using the default AWS Elastic Beanstalk domain, which does not have SSL/HTTPS configured. Future versions will include HTTPS support through a custom domain and SSL certificate.


## 🚀 Project Highlights

✅ Data Cleaning & Feature Engineering

✅ Exploratory Data Analysis (EDA)

✅ Feature Scaling using StandardScaler

✅ Ridge Regression Model

✅ Flask Web Application

✅ Model Serialization using Pickle

✅ AWS Elastic Beanstalk Deployment

✅ Real-Time Predictions

---

## 📌 Problem Statement

Forest fires can cause severe environmental and economic damage. Predicting the Fire Weather Index (FWI) helps estimate fire danger levels and supports better decision-making for fire prevention and management.

This project uses meteorological data to predict FWI through a Machine Learning model deployed as a web application.

---

## 🛠️ Tech Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Machine Learning     | Scikit-Learn          |
| Data Processing      | Pandas, NumPy         |
| Web Framework        | Flask                 |
| Model Storage        | Pickle                |
| Cloud Platform       | AWS Elastic Beanstalk |
| Version Control      | Git & GitHub          |

---

## 📊 Dataset Features

The model uses the following inputs:

* Temperature
* Relative Humidity (RH)
* Wind Speed (Ws)
* Rain
* FFMC
* DMC
* ISI
* Classes
* Region

Target Variable:

* Fire Weather Index (FWI)

---

## 🧠 Machine Learning Pipeline

```text
Data Collection
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Train-Test Split
       │
       ▼
Feature Scaling
       │
       ▼
Ridge Regression Model
       │
       ▼
Model Evaluation
       │
       ▼
Pickle Serialization
       │
       ▼
Flask Application
       │
       ▼
AWS Deployment
```

---

## 📂 Project Structure

```bash
Algerian_ForestFire_Predictor/
│
├── models/
│   ├── ridgeregression.pkl
│   └── Scaler.pkl
│
├── templates/
│   ├── home.html
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── notebooks/
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Souravs-Codes/Algerian_ForestFire_Predictor.git
```

### Navigate to Project Directory

```bash
cd Algerian_ForestFire_Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📈 Model Used

### Ridge Regression

Ridge Regression is a regularized version of Linear Regression that helps reduce overfitting by penalizing large coefficient values.

Benefits:

* Better generalization
* Reduced overfitting
* Handles multicollinearity
* Stable predictions

---

## 📸 Application Screenshots

### Home Page

(Add Screenshot Here)

### Prediction Page

(Add Screenshot Here)

---

## ☁️ Deployment

The application is deployed using:

* AWS Elastic Beanstalk
* Python Environment
* Flask Application
* Pickle Model Artifacts

This deployment allows users to access the prediction system directly through a web browser without installing any dependencies.

---

## 🔮 Future Improvements

* Random Forest Regressor
* XGBoost Regressor
* Better UI/UX
* Real-Time Weather API Integration
* Prediction History Tracking
* Docker Containerization
* CI/CD Pipeline

---

## 👨‍💻 Author

### Sourav Mukherjee

GitHub:
https://github.com/Souravs-Codes

---

## ⭐ Support

If you found this project useful, please consider giving it a star on GitHub.

Star ⭐ the repository to support the project.
