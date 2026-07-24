import pandas as pd

RAW_PATH = "./cars-data/raw/autos.csv"
CLEAN_PATH = "./cars-data/cleaned/autos-cleaned .csv"

KEEP_COLUMNS = [
    "price",
    "brand",
    "yearOfRegistration",
    "odometer",
    "powerPS",
    "notRepairedDamage",
]

REQUIRED_COLUMNS = [
    "price",
    "brand",
    "yearOfRegistration",
    "odometer",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "notRepairedDamage",
]

def is_falsy(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped == "nan"
    return value == 0

df = pd.read_csv(RAW_PATH, encoding="latin-1")

#drop columns that are not in KEEP_COLUMNS
df = df[KEEP_COLUMNS]

TRANSLATIONS = {
    "notRepairedDamage": {
        "nein": "no",
        "ja": "yes",
    },
}

#translate values in columns to English
for column, mapping in TRANSLATIONS.items():
    df[column] = df[column].replace(mapping)
    
# standardize values and clean price as a string
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype(str).str.lower()

    if column == "price":
        df[column] = df[column].str.replace(r"[$,]", "", regex=True).str.strip()
    elif column == "odometer":
        df[column] = df[column].str.replace(r"[,km]", "", regex=True).str.strip()

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

# drop rows with falsy values in required columns only
df = df[~df[REQUIRED_COLUMNS].apply(lambda row: any(is_falsy(value) for value in row), axis=1)]

# fill missing optional categoricals; treat 0 powerPS as missing
for column in OPTIONAL_CATEGORICAL_COLUMNS:
    df[column] = df[column].apply(lambda value: "unknown" if is_falsy(value) else value)

df.loc[df["powerPS"].isna() | (df["powerPS"] == 0), "powerPS"] = pd.NA

# remove rows with invalid registration year, powerPS, odometer values, and month of registration
MIN_REGISTRATION_YEAR = 1950
MAX_REGISTRATION_YEAR = 2016
MIN_POWERPS = 0
MAX_POWERPS = 1000
MIN_ODOMETER = 0
MAX_ODOMETER = 1000000
MIN_MONTH_OF_REGISTRATION = 1
MAX_MONTH_OF_REGISTRATION = 12

df = df[
    (df["price"] > 0)
    & (df["yearOfRegistration"] >= MIN_REGISTRATION_YEAR)
    & (df["yearOfRegistration"] <= MAX_REGISTRATION_YEAR)
    & (df["powerPS"].isna() | df["powerPS"].between(MIN_POWERPS, MAX_POWERPS))
    & (df["odometer"] >= MIN_ODOMETER)
    & (df["odometer"] <= MAX_ODOMETER)
]

# remove duplicates
df = df.drop_duplicates()

#save to csv
df.to_csv(CLEAN_PATH, index=False)

print(len(df))
print(df.head())
