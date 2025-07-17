# train_model.py
import pandas as pd
from feature_engineering import extract_wallet_features # type: ignore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_json("user-wallet-transactions.json")

# Feature Engineering
features_df = extract_wallet_features(df)

print("Available columns:", features_df.columns)
print(features_df.head())

# Create proxy label: total deposit amount in USD
features_df["label"] = features_df["total_deposits_usd"]

# Drop wallets with missing labels
features_df = features_df.dropna(subset=["label"])

X = features_df.drop(columns=["wallet", "label"])
y = features_df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "trained_model.pkl")
print("✅ Model trained and saved as trained_model.pkl")
