import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

COLUMNS = [
    "price",
    "brand",
    "yearORegistration",
    "odometer",
    "powerPS",
    "notRepairedDamage",
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

# keep the top 20 predictors using recursive feature elimination
selector = RFE(estimator=LinearRegression(), n_features_to_select=20)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
selected_features = feature_names[selector.support_]

# create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print intercept and coefficients and the evaluation metrics
print(f"Intercept: {model.intercept_}")
for feature, coefficient in zip(selected_features, model.coef_):
    print(f"{feature}: {coefficient}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
