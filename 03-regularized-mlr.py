import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

COLUMNS = [
    "price",
    "brand",
    "yearOfRegistration",
    "odometer",
    "powerPS",
    "notRepairedDamage"
]

df = pd.read_csv("./cars-data/cleaned/autos-cleaned .csv")

# convert the target to numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

# separate features and target/label variable
X = df[COLUMNS[1:]]
y = df[COLUMNS[0]]

# one-hot encode the categorical columns so linear regression can use them
categorical_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X = X.fillna(0).astype(float)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_names = X.columns

# scale features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

alphas = np.logspace(-2, 4, 100)

ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)


def print_model_results(name, model, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    non_zero_features = feature_names[model.coef_ != 0]

    print(f"\n{name}")
    print(f"Best alpha: {model.alpha_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Non-zero coefficients: {len(non_zero_features)}")
    for feature, coefficient in zip(feature_names, model.coef_):
        if coefficient != 0:
            print(f"{feature}: {coefficient}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


print_model_results("Ridge Regression", ridge, ridge_pred)
print_model_results("Lasso Regression", lasso, lasso_pred)
