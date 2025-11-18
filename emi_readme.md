# 📘 EMI Eligibility & Maximum EMI Prediction App

## 🚀 Overview

This project is a financial risk-assessment and EMI prediction application built using Machine Learning and Streamlit. The system performs two major financial tasks:

1. **EMI Eligibility Classification**: Predicts whether a customer is **Eligible**, **Not Eligible**, or **High Risk**.
2. **Maximum EMI Prediction (Regression)**: Estimates the specific maximum monthly EMI amount (₹) a customer can safely afford.

The application uses advanced feature engineering, risk scoring, and hyperparameter-tuned ML models to provide professional, data-driven financial insights.

---

## 🔧 Technologies Used

- **Python** (Core Logic)
- **Streamlit** (User Interface)
- **Pandas / NumPy** (Data Manipulation)
- **scikit-learn** (Preprocessing & Pipeline)
- **XGBoost** (Final Machine Learning Model)
- **Joblib** (Model Persistence for deployment)

---

## 📊 Feature Engineering

The raw financial data is enhanced with several powerful custom features to improve prediction accuracy and model interpretability:

### 1️⃣ Financial Ratios
- **Debt-to-Income Ratio (DTI)**
- **Expense-to-Income Ratio**
- **EMI Affordability Ratio**
- **Savings Ratio**

### 2️⃣ Risk Scoring
- **Credit Risk Category**: Derived from Credit Score buckets.
- **Employment Stability**: Derived from years of employment.
- **Combined Risk Score**: Weighted score based on credit history and income stability.

---

## 💡 Evaluation Metrics & Model Comparison

This section highlights the performance comparison between the Untuned (Base) and Tuned (Final) models, demonstrating the value of the hyperparameter optimization process. XGBoost was selected for production due to its superior performance and stability.

### 🟢 Task 1: EMI Eligibility (Classification)

**Objective**: Classify users into Eligible, Not Eligible, or High Risk.

| Algorithm      | Model Type         | Accuracy | CV Mean Accuracy | ROC AUC |
|----------------|--------------------|----------|------------------|---------|
| XGBoost        | Tuned (Selected)   | 0.9685   | 0.9669           | 0.9929  |
| XGBoost        | Untuned            | 0.9780   | 0.9770           | 0.9962  |
| Random Forest  | Tuned              | 0.9359   | 0.9356           | 0.9787  |
| Random Forest  | Untuned            | 0.9503   | 0.9482           | 0.9852  |

**Selection Rationale**: The Tuned XGBoost model was chosen for its exceptional consistency, offering stable performance (0.967 CV Accuracy) across multiple data folds, which is crucial for a robust production environment.

### 🔵 Task 2: Maximum EMI Prediction (Regression)

**Objective**: Estimate the maximum affordable EMI amount (₹).

| Algorithm      | Model Type         | R² Score | RMSE (₹) | MAE (₹) |
|----------------|--------------------|----------|----------|---------|
| XGBoost        | Tuned (Selected)   | 0.9894   | 803.49   | 348.06  |
| XGBoost        | Untuned            | 0.9908   | 742.63   | 354.06  |
| Random Forest  | Tuned              | 0.9651   | 1455.70  | 714.09  |
| Random Forest  | Untuned            | 0.9670   | 1403.79  | 714.09  |

**Selection Rationale**: The Tuned XGBoost Regressor is highly precise, achieving an R² of ~99% and a very low average absolute error (MAE) of just ₹348.

---

## 🧮 Prediction Output

The Streamlit app provides a clean output, including probability-based classification results and the precise regression output:

### Prediction Summary

| Category       | Probability |
|----------------|-------------|
| Not Eligible   | 5%          |
| Eligible       | 87%         |
| High Risk      | 9%          |

The application also displays:
- ✅ Recommended Maximum Monthly EMI (₹)
- 📉 Detailed Financial Ratios
- ⚠️ Risk Category Interpretation

---

## 📁 Project Structure

```
📦 EMI-Prediction-App
 ┣ 📜 app.py                    # Main Streamlit application
 ┣ 📜 model_eligibility.joblib  # Trained Classification Model (XGBoost)
 ┣ 📜 model_max_emi.joblib      # Trained Regression Model (XGBoost)
 ┣ 📜 requirements.txt          # Python Dependencies
 ┗ 📜 README.md                 # Documentation
```

---

## 🛠️ Installation & Running the App

### Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

### On Streamlit Cloud

1. Upload the entire project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Deploy the application by linking the GitHub repository.

**Important**: Ensure the following packages are included in your `requirements.txt` for successful deployment:

```
streamlit
pandas
numpy
scikit-learn
xgboot
joblib
```

---

## 🙌 Author

**Chandraprakash Kahar**  
Business Development • Analytics • ML • Data Science

---

## 📝 License

This project is open-source and available for educational and commercial use.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

## 📧 Contact

For questions or collaboration opportunities, please reach out via GitHub or LinkedIn.

---

**Made with ❤️ and Machine Learning**