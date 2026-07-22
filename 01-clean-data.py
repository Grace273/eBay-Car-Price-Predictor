import pandas as pd

RAW_PATH = "./cars-data/raw/autos.csv"
CLEAN_PATH = "./cars-data/cleaned/autos-cleaned .csv"

KEEP_COLUMNS = [
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

REQUIRED_COLUMNS = [
    "price",
    "brand",
    "model",
    "yearOfRegistration",
    "odometer",
]

df = pd.read_csv(RAW_PATH, encoding="latin-1")

#drop columns that are not in KEEP_COLUMNS
df = df[KEEP_COLUMNS]

#drop rows that have missing values in REQUIRED_COLUMNS
df = df.dropna(subset=REQUIRED_COLUMNS)

TRANSLATIONS = {
    "vehicleType": {
        "limousine": "sedan",
        "kleinwagen": "compact",
        "kombi": "wagon",
        "bus": "van",
        "cabrio": "convertible",
        "coupe": "coupe",
        "suv": "suv",
        "andere": "other",
    },
    "gearbox": {
        "manuell": "manual",
        "automatik": "automatic",
    },
    "fuelType": {
        "benzin": "gasoline",
        "diesel": "diesel",
        "lpg": "lpg",
        "cng": "cng",
        "hybrid": "hybrid",
        "elektro": "electric",
        "andere": "other",
    },
    "notRepairedDamage": {
        "nein": "no",
        "ja": "yes",
    },
}

#translate values in columns to English
for column, mapping in TRANSLATIONS.items():
    df[column] = df[column].replace(mapping)

# standardize column names to lowercase
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.lower()

# remove duplicates
df = df.drop_duplicates()

#save to csv
df.to_csv(CLEAN_PATH, index=False)

print(len(df))
print(df.head())
