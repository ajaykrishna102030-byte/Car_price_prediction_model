# 🚗 Car Price Prediction using Machine Learning

A complete end-to-end machine learning project that predicts the **selling price of used cars** based on features like make, model, engine size, mileage, fuel type, and transmission.

---

## 📁 Project Structure

```
car_price_prediction/
├── car_price_prediction.ipynb   # Jupyter Notebook (step-by-step)
├── car_price_prediction.py      # Python script version
├── Car_Price_Prediction.csv     # Dataset (place here before running)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── PROJECT_REPORT.docx          # Complete project report (Word)
```

---

## 📊 Dataset

| Detail | Info |
|--------|------|
| **File** | `Car_Price_Prediction.csv` |
| **Records** | 1000 rows |
| **Features** | 8 columns |
| **Target** | `Price` (USD) |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `Make` | Categorical | Car manufacturer (Honda, Ford, BMW, Audi, Toyota) |
| `Model` | Categorical | Car model (Model A–E) |
| `Year` | Integer | Year of manufacture (2000–2021) |
| `Engine Size` | Float | Engine displacement in Litres (1.0–4.5L) |
| `Mileage` | Integer | Total kilometres driven |
| `Fuel Type` | Categorical | Petrol / Diesel / Electric |
| `Transmission` | Categorical | Manual / Automatic |
| `Price` | Float (**Target**) | Selling price in USD |

---

## 🧠 Project Description

This project implements a full ML pipeline:

1. **Data Loading & Exploration** — shape, dtypes, missing values, statistics
2. **Exploratory Data Analysis (EDA)** — 7 visualisation charts covering price distribution, categorical breakdowns, scatter plots, correlation heatmap, and box plots
3. **Data Preprocessing** — duplicate removal, Label Encoding of categorical features
4. **Feature Engineering** — derived `Car_Age` from `Year` (2024 − Year)
5. **Train/Test Split** — 80/20 split with `StandardScaler` normalisation
6. **Model Training** — 4 regression models compared:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
7. **Evaluation** — MAE, RMSE, R² on held-out test set
8. **Feature Importance** — Random Forest feature importance bar chart
9. **Sample Predictions** — Side-by-side actual vs predicted prices

---

## 🏆 Results Summary

| Model | MAE (USD) | RMSE (USD) | R² Score |
|-------|-----------|------------|----------|
| Linear Regression | ~3,200 | ~4,100 | ~0.37 |
| Decision Tree | ~1,800 | ~2,600 | ~0.75 |
| Random Forest | ~1,500 | ~2,100 | ~0.84 |
| **Gradient Boosting** | **~1,400** | **~1,950** | **~0.86** |

> ✅ **Best Model:** Gradient Boosting Regressor — R² ≈ 0.86

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your dataset

Copy `Car_Price_Prediction.csv` into the `car_price_prediction/` folder.

### 3a. Run the Jupyter Notebook

```bash
jupyter notebook car_price_prediction.ipynb
```

### 3b. Run as Python script

```bash
python car_price_prediction.py
```

---

## 📦 Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.5.0 | Data loading & manipulation |
| `numpy` | ≥1.23.0 | Numerical operations |
| `matplotlib` | ≥3.6.0 | Plotting & visualisation |
| `seaborn` | ≥0.12.0 | Statistical visualisations & heatmaps |
| `scikit-learn` | ≥1.2.0 | ML models, preprocessing, metrics |
| `jupyter` | ≥1.0.0 | Interactive notebook environment |

---

## 📈 Output Files Generated

| File | Description |
|------|-------------|
| `eda_price_distribution.png` | Price histogram & box plot |
| `eda_categorical.png` | Make, Fuel Type, Transmission counts |
| `eda_avg_price_by_make.png` | Avg price per make |
| `eda_scatter.png` | Price vs Mileage & Engine Size |
| `eda_price_by_year.png` | Avg price trend over years |
| `eda_heatmap.png` | Correlation heatmap |
| `eda_boxplot_cat.png` | Price by Fuel Type & Transmission |
| `model_results.png` | Actual vs Predicted, Residuals, R² comparison |
| `feature_importance.png` | Random Forest feature importances |

---

## 👤 Author

**Ajay Krishna**

---

## 📄 License

This project is for educational purposes only.
