
# Predicting Diabetes Using Machine Learning

This project aims to build an end-to-end machine learning pipeline to predict diabetes based on patient diagnostic data. The pipeline includes preprocessing, visualization, outlier handling, feature scaling, model training, evaluation, and model saving. Multiple algorithms are compared to determine the best performer.

---

## 📊 Dataset

- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases  
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)  
- **Features**: 8 numerical features (e.g., Glucose, Blood Pressure, BMI, Age) + 1 binary target (`Outcome`)  
- **Samples**: 768 patient records

---

## 📌 Objective

To predict whether a person has diabetes (`Outcome = 1`) or not (`Outcome = 0`) using various supervised classification models and robust preprocessing techniques.

---

## 🧹 Data Preprocessing

- **Missing Values**:
  - Identified implicit missing values (e.g., zeros in `Glucose`, `BloodPressure`, etc.)
  - Imputed using both **median imputation** and **KNNImputer**
- **Outlier Detection**: Interquartile Range (IQR) clipping
- **Skewness Correction**: `log1p` transformation
- **Scaling**: `RobustScaler` (resilient to outliers)

---

## 📈 Visualizations

- Class distribution barplot  
- Pairplots and histograms  
- Correlation heatmap  
- Boxplots for outlier detection  

---

## 🤖 Models Implemented

| Model                  | Accuracy | ROC AUC | Remarks                            |
|-----------------------|----------|---------|-------------------------------------|
| Logistic Regression    | 71.4%    | 0.79    | Baseline model                      |
| Random Forest          | 72.7%    | 0.80    | Balanced performance, low overfitting |
| K-Nearest Neighbors    | 66.6%    | 0.72    | Simple but slower at prediction     |
| Decision Tree          | 70.1%    | 0.69    | Interpretable but prone to overfit  |
| AdaBoost               | 75.7%    | 0.81    | Improved generalization             |
| Gradient Boosting      | 73.5%    | 0.80    | Powerful ensemble, longer training  |
| XGBoost                | 71.8%    | 0.79    | Efficient, regularized boosting     |
| LightGBM               | *        | *       | Fastest, native handling of features|

> 🚧 Replace `*` with values if available.

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision / Recall / F1-score  
- ROC AUC  
- Confusion Matrix  

---

## 🧠 Libraries Used

```python
pandas, numpy, matplotlib, seaborn  
scikit-learn, xgboost, lightgbm  
joblib (for model saving)
```

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/diabetes-ml-prediction.git
cd diabetes-ml-prediction
pip install -r requirements.txt
python main.py  # or open notebook.ipynb
```

---

## 📂 File Structure

```plaintext
.
├── diabetes.csv
├── diabetes_cleaned.csv
├── models/
│   ├── model_logistic.joblib
│   ├── model_rf.joblib
├── plots/
│   ├── heatmap.png
│   ├── roc_curve.png
├── notebook.ipynb
├── main.py
└── README.md
```

---

## 💡 Key Learnings

- Medical datasets require strict cleaning  
- Model performance varies with preprocessing  
- Evaluation must go beyond accuracy  
- `joblib` helps in model persistence  

---

## 👤 Author

**Santosh Guntuku**  
`23B2158` | Mechanical Engineering, IIT Bombay  
Email: [your-email]  
GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 🏁 Future Work

- Grid search hyperparameter tuning  
- SMOTE for class imbalance  
- Streamlit/Flask app deployment  
- Incorporating advanced clinical features  
