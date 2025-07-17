This project builds a Credit Scoring Model for Aave V2 users based on their on-chain activity such as deposits, borrows, repayments, and liquidations. It applies machine learning to assign a credit score to each wallet, making it easier to assess lending reliability in the DeFi ecosystem.

📦 Contents
wallet_data.json – Raw transaction data in JSON format.

feature_engineering.py – Extracts user-level features from raw JSON.

model_training.py – Trains and saves a credit scoring model.

score_generator.py – Loads new data and generates credit scores.

wallet_scores.csv – Output scores for each wallet.

analysis.md – Insights on scoring results and behavior patterns.

README.md – Project overview and usage instructions.

Project Architecture:
JSON Input (wallet_data.json)
     │
     ▼
[Feature Engineering]
     │
     ▼
Structured Wallet Features (DataFrame)
     │
     ├──> [Model Training (Random Forest)] → trained_model.pkl
     │
     └──> [Scoring New Wallets] → wallet_scores.csv

     Machine Learning Model
Algorithm: Random Forest Classifier

Target: Synthetic score based on custom behavior rules

Feature Set:

Total deposits, borrows, repayments, liquidations

Asset diversity count

Active days

Transaction counts

Custom behavior ratios (e.g., repay/deposit)

How to Run
1. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, install manually:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib
2. Feature Engineering
bash
Copy
Edit
python feature_engineering.py
This will process the JSON and create a feature matrix per wallet.

3. Train the Model
bash
Copy
Edit
python model_training.py
Trains the model and creates trained_model.pkl.

4. Score New Wallets
bash
Copy
Edit
python score_generator.py
Generates scores and outputs them to wallet_scores.csv.

