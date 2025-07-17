import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from llm.strategy_generator import generate_strategy_code
from predictor import load_lstm_model, predict_next_lstm_return
from decision_maker import make_decision
from backtester import run_backtest

from news_summary import fetch_news, summarize_news
from llm.strategy_generator import call_llm  # if call_llm is defined there


st.set_page_config(page_title="📈 LLM-Driven Trading Assistant")

st.title("💹 LLM-Powered Stock Strategy Generator")

# Step 0: Choose stock
st.sidebar.header("📊 Select Stock")
preset_stocks = {
    "NIFTY 50 (India)": "^NSEI",
    "RELIANCE (India)": "RELIANCE.NS",
    "TCS (India)": "TCS.NS",
    "Apple (US)": "AAPL",
    "Tesla (US)": "TSLA",
    "Amazon (US)": "AMZN",
    "Bitcoin (USD)": "BTC-USD"
}

stock_name = st.sidebar.selectbox("Choose a stock:", list(preset_stocks.keys()))
ticker = preset_stocks[stock_name]

st.sidebar.write(f"📌 Ticker: `{ticker}`")

# User strategy prompt
strategy_prompt = st.text_area("🧠 Enter your strategy idea:", 
    "Use a momentum strategy with 20/50 SMA crossover and generate a 'Signal' column.")

if st.button("🔍 Run Strategy"):
    with st.spinner(f"Fetching {stock_name} data and generating strategy..."):
        # Load selected stock data
        df = yf.download(ticker, start="2018-01-01", end=None)

        # ✅ Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns.name = None
            df.columns = df.columns.get_level_values(0)

        # ✅ Select required columns for indicators
        df = df[['Close', 'High', 'Low', 'Volume']]

        # Step 1: LLM-generated strategy code
        strategy_code = generate_strategy_code(strategy_prompt)
        st.code(strategy_code, language='python')

        # Step 2: Run backtest
        result_df = run_backtest(strategy_code, df.copy())

        if result_df is not None and 'Signal' in result_df.columns:
            st.success(f"✅ Strategy executed on {stock_name}!")
            st.subheader(f"📊 Last 5 Signals for `{ticker}`")
            st.dataframe(result_df[['Close', 'Signal']].tail())

            st.subheader("📉 Strategy vs Market Performance")
            fig, ax = plt.subplots(figsize=(10, 4))
            result_df['Cumulative_Market'].plot(ax=ax, label="Market", linewidth=2)
            result_df['Cumulative_Strategy'].plot(ax=ax, label="Strategy", linewidth=2)
            ax.set_ylabel("Cumulative Returns")
            ax.set_title("Performance Comparison")
            ax.legend()
            st.pyplot(fig)

            # Step 3: Show Latest News & LLM Summary Automatically
            st.subheader("📰 Latest News and LLM Summary")

            with st.spinner(f"Fetching and analyzing news for {stock_name}..."):
                news_list = fetch_news(ticker)

                if news_list:
                    for i, article in enumerate(news_list, 1):
                        st.markdown(f"**{i}.** {article}")

                    summary = summarize_news(news_list, call_llm)
                    st.markdown("### 🧠 Summary:")
                    st.markdown(summary)
                else:
                    st.warning("No news found or failed to fetch news.")



            # Step 4: Predict next-day return using LSTM
            from predictor import train_lstm_model

            model, scaler = train_lstm_model(df)  # train dynamically on selected stock
            predicted = predict_next_lstm_return(model, scaler, df)
            st.subheader(f"📈 Predicted Next-Day Return: `{predicted:.2%}`")

            # Step 5: Recommend action
            action = make_decision(predicted)
            st.subheader(f"🟢 Recommended Action: `{action}`")

        else:
            st.error("❌ Strategy execution failed or 'Signal' column not found.")
