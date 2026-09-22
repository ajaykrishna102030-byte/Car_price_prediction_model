# =============================================================
# Car Price Prediction using Machine Learning
# Dataset: Car_Price_Prediction.csv
# Columns: Make, Model, Year, Engine Size, Mileage,
#          Fuel Type, Transmission, Price
# Target : Price (USD)
# Author : Ajay Krishna
# =============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

print("✅ All libraries imported successfully!\n")

# ── 2. Load & Explore Dataset ────────────────────────────────
df = pd.read_csv('Car_Price_Prediction.csv')
print(f"Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe()}")

print("\nUnique Makes       :", df['Make'].unique())
print("Unique Models      :", df['Model'].unique())
print("Unique Fuel Types  :", df['Fuel Type'].unique())
print("Unique Transmissions:", df['Transmission'].unique())
print(f"Year Range         : {df['Year'].min()} – {df['Year'].max()}")

# ── 3. Exploratory Data Analysis (EDA) ───────────────────────

# 3.1 Price Distribution & Box Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Price'], bins=30, color='steelblue', edgecolor='black')
axes[0].set_title('Car Price Distribution', fontsize=13)
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Frequency')
axes[1].boxplot(df['Price'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', color='navy'))
axes[1].set_title('Car Price Box Plot', fontsize=13)
axes[1].set_ylabel('Price (USD)')
plt.tight_layout()
plt.savefig('eda_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.2 Categorical Feature Counts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ['Make', 'Fuel Type', 'Transmission']):
    counts = df[col].value_counts()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='black')
    ax.set_title(f'{col} Distribution', fontsize=12)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('eda_categorical.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.3 Average Price by Make
avg_price = df.groupby('Make')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
avg_price.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Average Car Price by Make', fontsize=13)
plt.ylabel('Average Price (USD)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('eda_avg_price_by_make.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.4 Price vs Mileage & Engine Size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df['Mileage'], df['Price'], alpha=0.4, color='steelblue')
axes[0].set_title('Price vs Mileage', fontsize=12)
axes[0].set_xlabel('Mileage'); axes[0].set_ylabel('Price (USD)')
axes[1].scatter(df['Engine Size'], df['Price'], alpha=0.4, color='seagreen')
axes[1].set_title('Price vs Engine Size', fontsize=12)
axes[1].set_xlabel('Engine Size (L)'); axes[1].set_ylabel('Price (USD)')
plt.tight_layout()
plt.savefig('eda_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.5 Average Price by Year
avg_by_year = df.groupby('Year')['Price'].mean()
plt.figure(figsize=(12, 5))
plt.plot(avg_by_year.index, avg_by_year.values, marker='o', color='steelblue')
plt.title('Average Car Price by Year of Manufacture', fontsize=13)
plt.xlabel('Year'); plt.ylabel('Average Price (USD)')
plt.tight_layout()
plt.savefig('eda_price_by_year.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.6 Correlation Heatmap
plt.figure(figsize=(8, 6))
num_df = df[['Year', 'Engine Size', 'Mileage', 'Price']]
sns.heatmap(num_df.corr(), annot=True, fmt='.3f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap', fontsize=13)
plt.tight_layout()
plt.savefig('eda_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.7 Price by Fuel Type and Transmission
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.boxplot(column='Price', by='Fuel Type', ax=axes[0], boxprops=dict(color='steelblue'))
axes[0].set_title('Price by Fuel Type'); axes[0].set_xlabel('Fuel Type'); axes[0].set_ylabel('Price (USD)')
df.boxplot(column='Price', by='Transmission', ax=axes[1], boxprops=dict(color='coral'))
axes[1].set_title('Price by Transmission'); axes[1].set_xlabel('Transmission'); axes[1].set_ylabel('Price (USD)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('eda_boxplot_cat.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Data Preprocessing ────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed : {before - len(df)}")
print(f"Rows after cleaning: {len(df)}")

# Label Encode categorical columns
le = LabelEncoder()
cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"  Encoded '{col}' ✅")

# ── 5. Feature Engineering & Train/Test Split ────────────────
df_encoded['Car_Age'] = 2024 - df_encoded['Year']
df_encoded.drop(columns=['Year'], inplace=True)

print("\nFinal feature columns:")
print(df_encoded.columns.tolist())

X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining samples  : {X_train.shape[0]}")
print(f"Testing  samples  : {X_test.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# ── 6. Model Training ─────────────────────────────────────────
models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print(f"\n{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R² Score':>10}")
print('-' * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'model': model, 'preds': preds}
    print(f"{name:<25} {mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
print(f"\n✅ Best Model: {best_name}  (R² = {best['R2']:.4f})")

# ── 7. Model Evaluation Visualisations ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs Predicted
axes[0].scatter(y_test, best['preds'], alpha=0.5, color='steelblue',
                edgecolors='navy', linewidths=0.3)
lims = [min(y_test.min(), best['preds'].min()), max(y_test.max(), best['preds'].max())]
axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Price (USD)')
axes[0].set_ylabel('Predicted Price (USD)')
axes[0].set_title(f'Actual vs Predicted\n({best_name})', fontsize=11)
axes[0].legend()

# Residual Distribution
residuals = y_test.values - best['preds']
axes[1].hist(residuals, bins=30, color='coral', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (Actual - Predicted)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution', fontsize=11)

# R² Comparison Bar Chart
names  = list(results.keys())
r2s    = [results[k]['R2'] for k in names]
colors = ['seagreen' if n == best_name else 'steelblue' for n in names]
bars   = axes[2].bar(names, r2s, color=colors, edgecolor='black')
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel('R² Score')
axes[2].set_title('Model R² Comparison', fontsize=11)
axes[2].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, r2s):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_results.png")

# ── 8. Feature Importance ────────────────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importances – Random Forest Regressor', fontsize=13)
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_importance.png")

print("\nTop Features:")
print(importances.to_string())

# ── 9. Sample Prediction ──────────────────────────────────────
sample_preds  = best['model'].predict(X_test_sc[:5])
sample_actual = y_test.values[:5]

print(f"\nSample Predictions using: {best_name}\n")
print(f"{'#':<5} {'Actual Price':>15} {'Predicted Price':>17} {'Difference':>12}")
print('-' * 55)
for i, (actual, pred) in enumerate(zip(sample_actual, sample_preds), 1):
    diff = actual - pred
    print(f"{i:<5} ${actual:>14,.2f} ${pred:>16,.2f} ${diff:>11,.2f}")
