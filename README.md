# Boston Housing Price Prediction: Regression Model Comparison

## Objective
Predict median home values in the Boston housing dataset using multiple regression approaches, and compare scikit-learn implementations against custom-built models to deepen understanding of the underlying algorithms.

## Dataset
- **Source:** Boston Housing dataset (506 observations, 13 features)
- **Target:** MEDV — Median value of owner-occupied homes in $1,000s
- **Features:** Crime rate, zoning, proximity to employment, tax rate, pupil-teacher ratio, and more
- **Preprocessing:** Mean imputation for missing values, MinMaxScaler for distance-based models

## Methods

### Part 1: Model Comparison (scikit-learn + GridSearchCV)
| Model | Best Hyperparameter | MSE | MAE | MAPE |
|-------|-------------------|-----|-----|------|
| KNN Regressor | n_neighbors=11 | 45.01 | 4.32 | 0.194 |
| Decision Tree | max_depth=11 | 58.72 | 4.69 | 0.217 |
| Random Forest | max_depth=16, n_estimators=10 | 48.13 | 3.90 | 0.166 |

### Part 2: Linear Regression from Scratch
Implemented gradient descent optimization (1M iterations, lr=1e-6). Converged within ~6% relative divergence from scikit-learn's closed-form solution.

### Part 3: KNN Regression from Scratch
Implemented Euclidean distance-based KNN prediction. Produces **identical results** to scikit-learn (max difference = 0.0).

## Key Findings
- **Random Forest** achieved the best generalization (lowest MAPE at 16.6%)
- **Feature scaling** significantly improved KNN performance
- Custom implementations validated understanding of gradient descent and distance-based prediction

## Tools
Python, pandas, scikit-learn, matplotlib, NumPy

## Usage
```bash
# Clone and run
git clone https://github.com/tommypotts/boston-housing-regression.git
cd boston-housing-regression
jupyter notebook boston_housing_regression_analysis.ipynb
```

## Files
- `boston_housing_regression_analysis.ipynb` — Full analysis notebook
- `boston_housing_missing_values.csv` — Raw dataset (with missing values)
- `boston_housing.csv` — Cleaned dataset (after imputation)
