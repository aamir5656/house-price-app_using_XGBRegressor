
# 🏠 Housing Price Prediction Web App

This is a simple and professional Streamlit web application for predicting housing prices using machine learning models. The selected model for deployment is **XGBRegressor** based on performance and reduced overfitting.

---

## 📊 Model Comparison

Two models were trained on the dataset:

### 1. XGBRegressor
- **Test Accuracy:** 88.23%
- **Train Accuracy:** 91.52%
- **Overfitting:** Low

### 2. RandomForestRegressor
- **Test Accuracy:** 86.32%
- **Train Accuracy:** 96.89%
- **Overfitting:** High

Although both models showed similar test scores, `XGBRegressor` was chosen for the final app due to its better generalization and reduced overfitting.

---

## 🚀 Features
- Predict housing prices based on user inputs
- Clean and user-friendly interface
- Scikit-learn & XGBoost based model pipeline
- Deployed using Streamlit

---

## 💡 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Files Included

- `app.py`: Streamlit application script
- `xgb_model.pkl`: Trained XGBoost model
- `scaler.pkl`: Scaler used for preprocessing
- `requirements.txt`: List of required Python libraries

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

---

📂 Dataset Source
This project uses a publicly available Housing Prices Dataset originally sourced from Kaggle for educational purposes.

📝 Note: The dataset is used only for learning and practice purposes as part of a student project. All credit for the dataset goes to its original creators on Kaggle.

## ✨ Author

Aamir Shahzad – Housing Price Prediction Project (2025)

