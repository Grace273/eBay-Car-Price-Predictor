import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("./cars-data/cleaned/autos-cleaned .csv")

#Separate features and target variable
COLUMNS = [
    "price",
    "brand",
    "model",
    "yearOfRegistration",
    "odometer",
    "powerPS",
    "vehicleType",
    "gearbox",
    "fuelType",
    "notRepairedDamage",
    "monthOfRegistration",
]

X = df[COLUMNS[1:]]
y = df[COLUMNS[0]]

#One-hot encode the categorical columns so random forest can use them
categorical_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X = X.fillna(0).astype(float)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("1. Total features created:", X.shape[1])
print("2. Rows with 0 in yearOfRegistration:", (X['yearOfRegistration'] == 0).sum())
print("3. Top 5 most common columns:\n", X.columns[:5].tolist())