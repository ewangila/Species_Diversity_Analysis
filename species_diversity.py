import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from faraway.datasets import galapagos
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")

# 1. Load & Prepare Data
data = galapagos.load()
X_predictors = data[['Area', 'Elevation', 'Nearest']]
X = sm.add_constant(X_predictors)
y = data['Species']

# 2. Fit OLS Model
model = sm.OLS(y, X).fit()
print(model.summary())

# 3. Multicollinearity Check
vif_df = pd.DataFrame({
    'Feature': X_predictors.columns,
    'VIF': [variance_inflation_factor(X_predictors.values, i) for i in range(X_predictors.shape[1])]
})
print("\nVariance Inflation Factors")
print(vif_df)

plt.figure(figsize=(6, 4))
sns.heatmap(X_predictors.corr(), annot=True, cmap='YlGnBu', vmin=-1, vmax=1)
plt.title("Predictor Correlation Heatmap")
plt.show()

# 4. Residual Diagnostics
residuals, fitted_vals = model.resid, model.fittedvalues
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].scatter(fitted_vals, residuals, alpha=0.7, edgecolors='k')
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')

axes[0, 1].hist(residuals, bins=10, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Histogram of Residuals')

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

axes[1, 1].scatter(fitted_vals, np.sqrt(np.abs(residuals / residuals.std())), alpha=0.7, edgecolors='k')
axes[1, 1].set_title('Scale-Location')

plt.tight_layout()
plt.show()

stat, p_val = stats.shapiro(residuals)
print(f"\nResidual Tests")
print(f"Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p_val:.4f}")

# 5. Model Validation & Comparison
cv_scores = cross_val_score(LinearRegression(), X_predictors, y, cv=5, scoring='r2')
print(f"\nModel Validation")
print(f"5-Fold CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

model_simple = sm.OLS(y, sm.add_constant(data[['Area']])).fit()
print(f"Simple Model (Area only) R²: {model_simple.rsquared:.4f}")
print(f"Full Model R²:               {model.rsquared:.4f}")

# 6. Prediction with Intervals
new_island = pd.DataFrame({'const': [1.0], 'Area': [10.0], 'Elevation': [100.0], 'Nearest': [1.5]})
pred_frame = model.get_prediction(new_island).summary_frame(alpha=0.05)
print("\nPrediction (Area=10, Elevation=100, Nearest=1.5)")
print(pred_frame[['mean', 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']])