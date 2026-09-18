# Galápagos Islands Species Diversity Analysis

Multiple linear regression analysis of plant species richness across the Galápagos Islands.  
This project models the relationship between island characteristics (**Area**, **Elevation**, and **Nearest** island distance) and the number of plant **Species**.

## Objective

Identify the key ecological factors that drive plant species richness and evaluate the quality of a multiple linear regression model through:

- Multicollinearity diagnostics (VIF & correlation heatmap)
- Residual analysis (residuals vs fitted, histogram, Q-Q plot, scale-location)
- Formal normality testing (Shapiro–Wilk)
- Model validation via 5-fold cross-validation
- Comparison against a simple baseline model (Area only)
- Prediction with confidence and prediction intervals

## Dataset

The analysis uses the classic **Galápagos** dataset from the `faraway` package, which contains species counts and geographic/ecological variables for 30 islands.

| Variable   | Description                          |
|------------|--------------------------------------|
| Species    | Number of plant species              |
| Area       | Island area (km²)                    |
| Elevation  | Maximum elevation (m)                |
| Nearest    | Distance to nearest island (km)      |
| Scruz      | Distance to Santa Cruz (km)          |
| Adjacent   | Area of adjacent island (km²)        |

## Project Structure
```
Species_Diversity_Analysis/
├── Species_Diversity_Analysis.ipynb   # Full analysis notebook
├── species_diversity.py               # Standalone Python script
├── requirements.txt
├── .gitignore
└── LICENSE
```
## Requirements

- Python 3.8+
- Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```
## How to Run

Option 1 – Jupyter Notebook (recommended)
```Bash
jupyter notebook Species_Diversity_Analysis.ipynb
```

Option 2 – Python script
```Bash
python species_diversity.py
```
## Methods Summary

1. **Data preparation** – Load Galápagos data and construct the design matrix with intercept.
2. **Multicollinearity check** – Compute Variance Inflation Factors (VIF) and correlation heatmap.
3. **Model fitting** – Ordinary Least Squares (OLS) via `statsmodels`.
4. **Residual diagnostics** – Visual and formal checks for linearity, normality, and homoscedasticity.
5. **Validation** – 5-fold cross-validated R² and comparison with a simple Area-only model.
6. **Prediction** – Generate mean predictions with 95% confidence and prediction intervals for a hypothetical new island.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
