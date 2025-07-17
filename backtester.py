def run_backtest(code, df):
    import numpy as np
    import pandas as pd

    local_vars = {
        'df': df.copy(),
        'np': np,
        'pd': pd
    }

    try:
        exec(code, {}, local_vars)
    except Exception as e:
        print("Strategy execution failed:", e)
        return None

    df = local_vars.get('df')
    if df is None or 'Signal' not in df.columns:
        return None

    # Add returns column
    df['Returns'] = df['Close'].pct_change()

    # Convert Buy/Sell signals into positions (1 for Buy, -1 for Sell, 0 for Hold)
    df['Position'] = df['Signal'].map({'Buy': 1, 'Sell': -1, 'Hold': 0}).fillna(0)

    # Strategy daily returns
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)

    # Cumulative returns
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

    return df
