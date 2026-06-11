# 📈 AI Stock Price Predictor

An LSTM-based deep learning app that predicts future stock prices 
using real-time market data and technical indicators.

## 🧠 How It Works
- Fetches real-time stock data via yfinance 
- Engineers technical indicators: SMA, EMA, RSI, MACD
- Trains a 2-layer LSTM neural network on historical data
- Predicts future prices for any date you choose
- Displays UP/DOWN trend table + interactive chart

## 🛠️ Tech Stack
- Python, TensorFlow/Keras, Scikit-learn
- yfinance (real-time data)
- Streamlit (web app)
- Pandas, NumPy, Matplotlib

## 📊 Features
- Enter any stock symbol (AAPL, MSFT, GOOGL, etc.)
- Select any future date
- Get day-by-day price predictions
- Visual chart: Historical + AI predicted prices
- Color-coded trend table (Green=UP, Red=DOWN)
- Key metrics: Start Price, End Price, % Change

## ⚙️ Run Locally
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure
├── app.py                      # Main Streamlit app
├── Stock_Predictor_LSTM.ipynb  # Jupyter notebook
├── requirements.txt
└── README.md
