
---

# 💡 Bank Marketing Term Deposit Prediction

This repository contains the final project for the ADA442 Statistical Learning course. The goal is to predict whether a customer will subscribe to a term deposit based on marketing data collected by a Portuguese bank.

## 📊 Project Overview

This project uses the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), which contains data collected from direct phone call marketing campaigns. Each record represents a customer profile and the outcome of the marketing effort (subscribed or not).

We applied a complete machine learning pipeline including:

* Data cleaning and preprocessing
* Feature selection and engineering
* Model comparison (Logistic Regression, Random Forest, Neural Network)
* Hyperparameter tuning
* Model evaluation
* Deployment using Streamlit

## 🚀 Live Demo

🔗 **Streamlit App:** [Here](https://ada442-bankmarketingclassifier.streamlit.app)


## 📁 Project Structure

```
├── project.ipynb               # Jupyter Notebook with all code and analysis
├── streamlit_app.py           # Streamlit deployment script
├── optimized_rf_model.pkl 
├── input_columns.pkl           # Models
├── label_encoders.pkl
├── scaler.pkl
├── requirements.txt           # Required Python packages for deployment
├── bank-additional.csv        # Dataset (10% sample of full dataset)
```

## 🧠 Models Used

We compared several models and selected the best one based on F1 score after applying SMOTE for class imbalance:

* Logistic Regression
* Random Forest
* Multilayer Perceptron (MLP)

The final deployed model was fine-tuned using **Optuna** and evaluated on unseen test data.

## 📈 Performance Metrics

The selected model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

## ⚙️ How to Run Locally

1. Clone the repository
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run streamlit_app.py
   ```

## 📚 References

* Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31.
* Scikit-learn Documentation
* Streamlit Documentation
* Towards Data Science articles on ML pipelines and deployment

---


