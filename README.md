# 🔥 Algerian Forest Fire Predictor

An end-to-end **Machine Learning application** that predicts the **Fire Weather Index (FWI)** using meteorological and environmental data from the Algerian Forest Fires dataset.

The project covers the complete Machine Learning workflow, including:

- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Feature scaling
- Model training
- Model evaluation
- Model serialization
- Interactive web application
- Cloud deployment using Streamlit

---

## 🌐 Live Demo

🚀 **Try the application here:**

👉 https://algerianforestfirepredictor-hkpc5h2hxdsdwdjaso72hh.streamlit.app/

The application allows users to enter weather and environmental parameters and receive a predicted **Fire Weather Index (FWI)** in real time.

---

## 🔥 About the Project

Forest fires can cause significant environmental, economic, and ecological damage.

The **Fire Weather Index (FWI)** is an important indicator used to estimate fire danger based on weather and environmental conditions.

This project uses Machine Learning to predict the FWI from various meteorological parameters.

The trained model is integrated into a **Streamlit web application**, allowing users to interact with the model directly through a browser.

---

## 🎯 Problem Statement

The objective of this project is to build a Machine Learning model capable of predicting the **Fire Weather Index (FWI)** from weather conditions.

Given environmental inputs such as:

- Temperature
- Relative Humidity
- Wind Speed
- Rainfall
- FFMC
- DMC
- ISI
- Region

the model predicts the corresponding **FWI value**.

---

## 🚀 Project Highlights

- 🔥 Fire Weather Index prediction
- 🧹 Data cleaning and preprocessing
- 📊 Exploratory Data Analysis
- ⚙️ Feature engineering
- 📏 Feature scaling using `StandardScaler`
- 🤖 Ridge Regression model
- 💾 Model serialization using Pickle
- 🌐 Interactive Streamlit application
- ☁️ Cloud deployment using Streamlit
- ⚡ Real-time prediction

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Regression Model | Ridge Regression |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Feature Scaling | StandardScaler |
| Model Serialization | Pickle |
| Web Application | Streamlit |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📊 Dataset

The project uses the **Algerian Forest Fires Dataset**.

The dataset contains meteorological and environmental measurements collected from two regions of Algeria.

### Input Features

| Feature | Description |
|---|---|
| Temperature | Temperature of the environment |
| RH | Relative Humidity |
| Ws | Wind Speed |
| Rain | Amount of rainfall |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| ISI | Initial Spread Index |
| Classes | Fire / Not Fire classification |
| Region | Geographic region |

### Target Variable

**FWI — Fire Weather Index**

The FWI value represents the potential fire danger based on the given environmental conditions.

---

## 🧠 Machine Learning Pipeline

```text
              Dataset
                  │
                  ▼
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
        (StandardScaler)
                  │
                  ▼
          Ridge Regression
                  │
                  ▼
          Model Evaluation
                  │
                  ▼
       Model Serialization
             (Pickle)
                  │
                  ▼
       Streamlit Application
                  │
                  ▼
        Real-Time Prediction
