# app.py - FINAL ATTRACTIVE & WORKING VERSION
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(page_title="AI Stock Predictor", layout="wide", initial_sidebar_state="expanded")

# CUSTOM CSS - PROFESSIONAL LOOK
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3d72, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff512f, #dd2476);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.1rem;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,81,47,0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,81,47,0.6);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-table {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<h1 class="main-header">AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter any stock & future date → Get <b>accurate AI prediction</b> with trends!</p>', unsafe_allow_html=True)

# INPUTS
col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("**Stock Symbol**", "AMZN", help="e.g., AAPL, MSFT, GOOGL").upper().strip()
with col2:
    future_date = st.date_input("**Future Date**", datetime(2025, 11, 25))

# PREDICT BUTTON
if st.button("Predict Now", type="primary"):
    with st.spinner("Training AI model on latest data..."):
        try:
            # DATA
            raw = yf.download(ticker, start="2010-01-01", progress=False)
            if raw.empty:
                st.error("No data found!")
                st.stop()

            close_col = 'Close' if 'Close' in raw.columns else 'close'
            data = pd.DataFrame(raw[close_col]).copy()
            data.columns = ['Close']

            # FEATURES
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data.dropna(inplace=True)

            features = ['Close', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal']
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data[features])

            # SEQUENCES
            def seq(data, length=60):
                X, y = [], []
                for i in range(length, len(data)):
                    X.append(data[i-length:i])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)
            X, y = seq(scaled)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # MODEL
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(60, len(features))))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile('adam', 'mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=10)], verbose=0)

            # FUTURE PREDICTION (FIXED SCALE)
            last_date = data.index[-1].date()
            current_price = data['Close'].iloc[-1]
            days = (pd.to_datetime(future_date) - pd.to_datetime(last_date)).days
            if days <= 0:
                st.error("Future date must be after today!")
                st.stop()

            seq = scaled[-60:].copy()
            preds = []
            for _ in range(days):
                p = model.predict(seq.reshape(1, 60, len(features)), verbose=0)[0, 0]
                preds.append(p)

                # PROPER SCALED UPDATE
                close_seq = seq[:, 0]
                new_sma10 = np.mean(close_seq[-9:]) if len(close_seq) >= 9 else p
                new_sma50 = np.mean(close_seq[-49:]) if len(close_seq) >= 49 else p
                a12 = 2/(12+1); new_ema12 = a12 * p + (1-a12) * seq[-1, 3]
                a26 = 2/(26+1); new_ema26 = a26 * p + (1-a26) * seq[-1, 4]
                new_rsi = 50
                new_macd = new_ema12 - new_ema26
                a9 = 2/(9+1); new_sig = a9 * new_macd + (1-a9) * seq[-1, 7]

                new_row = np.array([p, new_sma10, new_sma50, new_ema12, new_ema26, new_rsi, new_macd, new_sig]).reshape(1, -1)
                new_row_scaled = scaler.transform(new_row)
                seq = np.vstack([seq[1:], new_row_scaled])

            prices = scaler.inverse_transform(
                np.concatenate([np.array(preds).reshape(-1,1), np.zeros((len(preds), len(features)-1))], axis=1)
            )[:, 0]

            dates = pd.bdate_range(start=last_date + timedelta(1), end=future_date)
            prices = prices[:len(dates)]

            # RESULTS
            st.success(f"Prediction: {last_date} → {future_date}")

            # TABLE
            df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in dates],
                'Price': [f"${x:.2f}" for x in prices],
                'Trend': ['UP' if i==0 else ('UP' if prices[i]>prices[i-1] else 'DOWN') for i in range(len(prices))]
            })
            df['Trend'] = df['Trend'].replace({'UP': 'UP', 'DOWN': 'DOWN'})
            st.markdown("<div class='prediction-table'>", unsafe_allow_html=True)
            st.dataframe(df.style.apply(lambda x: ['background: #d4edda' if v=='UP' else 'background: #f8d7da' for v in x], subset=['Trend']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # METRICS
            change = prices[-1] - current_price
            pct = (change / current_price) * 100
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.markdown(f"<div class='metric-card'><h3>Start Price</h3><h2>${current_price:.2f}</h2></div>", unsafe_allow_html=True)
            with colm2:
                st.markdown(f"<div class='metric-card'><h3>End Price</h3><h2>${prices[-1]:.2f}</h2></div>", unsafe_allow_html=True)
            with colm3:
                color = "green" if change > 0 else "red"
                st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, {'#11998e' if change>0 else '#ee5a52'} 0%, {'#38ef7d' if change>0 else '#ff6b6b'} 100%)'><h3>Change</h3><h2 style='color:white'>{change:+.2f} ({pct:+.2f}%)</h2></div>", unsafe_allow_html=True)

            # GRAPH
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data.index[-100:], data['Close'][-100:], label='Historical', color='#1e3d72', linewidth=2.5)
            ax.plot(dates, prices, label='AI Prediction', color='#ff512f', linestyle='--', linewidth=3)
            ax.axvline(last_date, color='green', linestyle=':', linewidth=2, label='Today')
            ax.set_title(f"{ticker} Stock Price: Past + AI Future Prediction", fontsize=18, fontweight='bold', color='#1e3d72')
            ax.set_ylabel("Price ($)", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")
