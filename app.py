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

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Price Predictor (LSTM + SMA + RSI + MACD)")
st.write("Enter stock and future date → Get accurate prediction!")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Symbol", "AAPL").upper()
with col2:
    future_date = st.date_input("Future Date", datetime(2025, 11, 25))

if st.button("Predict Now", type="primary"):
    with st.spinner("Training model on latest data..."):
        # === DATA ===
        data = yf.download(ticker, start="2010-01-01", progress=False)[['Close']]
        
        # === FEATURES ===
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

        # === SEQUENCES ===
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled, 60)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # === MODEL ===
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(60, len(features))))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                  callbacks=[early_stop], verbose=0)

        # === FUTURE PREDICTION (FIXED SCALE) ===
        last_date = data.index[-1].date()
        current_price = data['Close'][-1]
        days = (pd.to_datetime(future_date) - pd.to_datetime(last_date)).days

        if days <= 0:
            st.error("Future date must be after today!")
        else:
            seq = scaled[-60:].copy()
            preds = []
            for _ in range(days):
                p = model.predict(seq.reshape(1, 60, len(features)), verbose=0)[0, 0]
                preds.append(p)

                # UPDATE FEATURES IN SCALED SPACE (Fixed for no explosion)
                close_seq = seq[:, 0]
                new_sma10 = np.mean(close_seq[-9:]) if len(close_seq) >= 9 else p
                new_sma50 = np.mean(close_seq[-49:]) if len(close_seq) >= 49 else p
                alpha12 = 2/(12+1); new_ema12 = alpha12 * p + (1-alpha12) * seq[-1, 3]
                alpha26 = 2/(26+1); new_ema26 = alpha26 * p + (1-alpha26) * seq[-1, 4]
                new_rsi = 50
                new_macd = new_ema12 - new_ema26
                alpha9 = 2/(9+1); new_sig = alpha9 * new_macd + (1-alpha9) * seq[-1, 7]

                new_row = np.array([p, new_sma10, new_sma50, new_ema12, new_ema26, new_rsi, new_macd, new_sig]).reshape(1, -1)
                new_row_scaled = scaler.transform(new_row)
                seq = np.vstack([seq[1:], new_row_scaled])

            prices = scaler.inverse_transform(np.concatenate([np.array(preds).reshape(-1,1), 
                                                             np.zeros((len(preds), len(features)-1))], axis=1))[:, 0]
            dates = pd.bdate_range(start=last_date + timedelta(1), end=future_date)
            prices = prices[:len(dates)]

            # === RESULTS ===
            st.success(f"Prediction: {last_date} → {future_date}")
            df = pd.DataFrame({'Date': dates.strftime('%Y-%m-%d'), 'Price': [f"${x:.2f}" for x in prices]})
            st.table(df)

            change = prices[-1] - current_price
            st.metric("Overall Change", f"${change:+.2f}", f"{(change/current_price)*100:+.2f}%")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index[-100:], data['Close'][-100:], label='Past', color='blue')
            ax.plot(dates, prices, label='Future', color='red', linestyle='--')
            ax.axvline(last_date, color='green', linestyle=':', label='Today')
            ax.set_title(f"{ticker} Stock Prediction")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
