import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA 
# =========================
def load_data():
    """
    Load raw dataset from LoanDetection_raw folder
    Path dibuat dinamis agar aman di local & GitHub Actions
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_path = os.path.join(
        base_dir,
        "LoanDetection_raw",
        "LoanPrediction_raw.csv"
    )

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    return pd.read_csv(raw_path)


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):
    """
    Preprocessing sesuai notebook:
    - Drop Loan_ID
    - Encode target
    - One-hot encoding categorical features
    - Scaling numeric features
    """

    # Drop ID column
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Encode target (Y/N -> 1/0)
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Categorical features (encode, NOT scale)
    categorical_cols = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Numeric features (scale)
    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


# =========================
# SAVE OUTPUT
# =========================
def save_data(df):
    """
    Save cleaned dataset ke folder preprocessing
    """
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "LoanDetection_preprocessing"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "LoanPrediction_preprocessing.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"Preprocessing completed. File saved at: {output_path}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    df = load_data()
    df_clean = preprocess_data(df)
    save_data(df_clean)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
