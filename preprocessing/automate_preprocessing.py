import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===== CONFIG  =====
CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area"
]

NUMERICAL_COLS = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History"
]

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

OUTLIER_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def handle_outlier_iqr_clip(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"N": 0, "Y": 1})
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    cols_to_scale = [c for c in NUMERICAL_COLS if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    input_path = "/Users/zakialfadilah/Eksperimen_SML_MuhammadZakiAlfadilah/LoanDetection_raw/LoanPrediction_raw.csv"
    output_path = "/Users/zakialfadilah/Eksperimen_SML_MuhammadZakiAlfadilah/preprocessing/LoanDetection_preprocessing/LoanPrediction_preprocessing.csv"

    df = load_data(input_path)
    df = drop_id(df)
    df = handle_missing_values(df)
    df = handle_outlier_iqr_clip(df)
    df = encode_target(df)
    df = encode_categorical_features(df)
    df = scale_numerical_features(df)

    # validasi 
    if df[TARGET_COL].isna().any():
        raise ValueError("Target masih ada NaN setelah encoding. Cek nilai target selain 'Y'/'N'.")

    save_data(df, output_path)
    print(f"Preprocessing selesai. Output: {output_path}")


if __name__ == "__main__":
    main()
