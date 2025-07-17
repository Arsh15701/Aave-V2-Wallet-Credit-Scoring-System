import json
import pandas as pd
from feature_engineering import extract_wallet_features
import joblib

def load_transactions(json_path: str) -> pd.DataFrame:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    df = pd.json_normalize(raw_data)
    return df

def main():
    json_path = r"C:\Users\arsh0\Desktop\CSP\user-wallet-transactions.json"  # Ensure this file is in the same directory
    model_path = r"C:\Users\arsh0\Desktop\CSP\trained_model.pkl"

    print("🔄 Loading transaction data...")
    df = load_transactions(json_path)

    print("Columns in DataFrame:", df.columns.tolist())

    print(f"✅ Loaded {len(df)} transactions")

    print("🔧 Extracting features...")
    wallet_features = extract_wallet_features(df)

    print("📦 Loading trained model...")
    model = joblib.load(model_path)

    print("🧮 Scoring wallets...")
    X = wallet_features.drop(columns=['wallet'])  # Remove 'wallet' column
    scores = model.predict(X)                     # Predict using only feature columns


    result_df = wallet_features.copy()
    result_df['credit_score'] = scores

    output_path = "wallet_scores.csv"
    result_df.to_csv(output_path, index=False)

    print(f"✅ Scoring complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
