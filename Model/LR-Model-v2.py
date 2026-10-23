# 1. Required libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 2. Data Reading
df = pd.read_csv(r"C:\Users\dell\Desktop\Real Estate Price Prediction(LR)\Dataset\Real estate.csv")

# 3. Data preprocessing
print(df.isnull().sum())

print("\nDataset Info")
print(df.info())

X = df.drop(["Y house price of unit area", "No"], axis=1)
y = df["Y house price of unit area"]

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Baseline (Linear Regression) with Cross Validation
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
base_cv = cross_val_score(baseline, X_train, y_train, cv=5, scoring="r2")
print("\nLinear Regression CV R2:", round(base_cv.mean(), 3))

# 6. Model Comparison + Hyperparameter Tuning
candidates = {
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 6, 10],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
    ),
}

best_model = None
best_name = ""
best_cv_score = -np.inf

for name, (estimator, params) in candidates.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
    grid = GridSearchCV(pipe, params, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\n{name} -> CV R2: {grid.best_score_:.3f}")
    print("Best params:", grid.best_params_)
    if grid.best_score_ > best_cv_score:
        best_cv_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name

print("\nSelected Model:", best_name)

# 7. Prediction
y_pred = best_model.predict(X_test)

# 8. Model Performance
print("\nModel Performance")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 9. Save Model
with open("real_estate_model_v2.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\nModel saved successfully!")