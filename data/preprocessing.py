import yfinance as yf
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

def prepare_data(include_today=True, look_back=60):
    ticker = 'meta'
    period = '2y'
    interval = '1h'

    data = yf.download(ticker, period=period, interval=interval)
    if not include_today:
        data = data[:-1]
    if len(data) < 100:
        print(f"Se requieren al menos 180 precios históricos. Actualmente hay {len(data)} datos.")
        return None

    close_series = data['Close'].astype(float)
    volume_series = data['Volume'].astype(float)

    # Asegúrate de que las series sean 1D
    data['MACD'] = ta.trend.macd(close_series.squeeze())
    data['RSI'] = ta.momentum.rsi(close_series.squeeze())
    data['BB_upper'] = ta.volatility.bollinger_hband(close_series.squeeze())
    data['BB_lower'] = ta.volatility.bollinger_lband(close_series.squeeze())
    data['Volume_norm'] = MinMaxScaler().fit_transform(volume_series.values.reshape(-1, 1))
    data = data.dropna()

    feature_columns = ['Close', 'MACD', 'RSI', 'BB_upper', 'BB_lower', 'Volume_norm']
    scaler = MinMaxScaler((0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])

    def create_dataset(dataset, look_back):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i + look_back, :])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, look_back)
    return {"X": X, "y": y}, scaler
