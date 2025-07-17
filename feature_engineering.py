import pandas as pd
from collections import defaultdict

def compute_usd_value(row):
    try:
        return float(row['actionData.amount']) * float(row['actionData.assetPriceUSD'])
    except:
        return 0.0

def extract_wallet_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['usd_value'] = df.apply(compute_usd_value, axis=1)

    grouped = defaultdict(lambda: {
        'wallet': None,
        'num_deposits': 0,
        'total_deposits_usd': 0,
        'num_borrows': 0,
        'total_borrows_usd': 0,
        'num_repays': 0,
        'total_repays_usd': 0,
        'num_liquidations': 0,
        'unique_assets': set(),
        'timestamps': []
    })

    for _, row in df.iterrows():
        wallet = row.get('userWallet')
        action = row.get('action')
        usd_val = row.get('usd_value', 0)
        action_data = row.get('actionData', {})
        asset = action_data.get('assetSymbol', '') if isinstance(action_data, dict) else ''

        grouped[wallet]['wallet'] = wallet
        grouped[wallet]['timestamps'].append(row['timestamp'])
        grouped[wallet]['unique_assets'].add(asset)

        if action == 'deposit':
            grouped[wallet]['num_deposits'] += 1
            grouped[wallet]['total_deposits_usd'] += usd_val
        elif action == 'borrow':
            grouped[wallet]['num_borrows'] += 1
            grouped[wallet]['total_borrows_usd'] += usd_val
        elif action == 'repay':
            grouped[wallet]['num_repays'] += 1
            grouped[wallet]['total_repays_usd'] += usd_val
        elif action == 'liquidationcall':
            grouped[wallet]['num_liquidations'] += 1

    feature_rows = []
    for wallet, data in grouped.items():
        repay_ratio = (
            data['total_repays_usd'] / data['total_borrows_usd']
            if data['total_borrows_usd'] > 0 else 0
        )
        active_days = (
            (max(data['timestamps']) - min(data['timestamps'])).days + 1
            if data['timestamps'] else 0
        )

        feature_rows.append({
            'wallet': wallet,
            'total_deposits_usd': data['total_deposits_usd'],
            'total_borrows_usd': data['total_borrows_usd'],
            'total_repays_usd': data['total_repays_usd'],
            'repay_ratio': repay_ratio,
            'num_liquidations': data['num_liquidations'],
            'unique_assets_used': len(data['unique_assets']),
            'activity_days': active_days,
            'num_tx': len(data['timestamps'])
        })

    return pd.DataFrame(feature_rows)
