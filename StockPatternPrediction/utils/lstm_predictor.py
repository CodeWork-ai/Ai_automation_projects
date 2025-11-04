import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 3])  # Close price is at index 3
    return np.array(X), np.array(y)


def train_lstm_predictor(df, seq_length=50, epochs=30, batch_size=32):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    split = int(len(scaled_features) * 0.8)
    train_data = scaled_features[:split]
    test_data = scaled_features[split - seq_length:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    model = Sequential()
    model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping and learning rate reduction
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, lr_reduce],
        verbose=1
    )

    # Predictions
    predictions_scaled = model.predict(X_test)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df['Close'].values.reshape(-1, 1)
    close_scaler.fit(close_prices)

    predictions = close_scaler.inverse_transform(predictions_scaled)
    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_inv, predictions)
    mse = mean_squared_error(y_test_inv, predictions)
    r2 = r2_score(y_test_inv, predictions)

    # Accuracy style metric (pseudo accuracy for regression)
    accuracy = max(0, 100 - (mae / np.mean(y_test_inv)) * 100)

    test_dates = df['Date'].iloc[split - seq_length:].reset_index(drop=True)
    pred_dates = test_dates.iloc[seq_length:].reset_index(drop=True)

    prediction_df = pd.DataFrame({
        'Date': pred_dates,
        'Predicted Close Price': predictions.flatten()
    })

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2 Score': r2,
        'Approx Accuracy (%)': accuracy
    }

    return model, scaler, metrics, prediction_df


def predict_future(model, df, scaler, seq_length=50, future_days=7):
    last_sequence = df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-seq_length:]
    scaled_seq = scaler.transform(last_sequence)
    prediction_input = scaled_seq.reshape(1, seq_length, scaled_seq.shape[1])

    predictions_scaled = []
    for _ in range(future_days):
        pred = model.predict(prediction_input, verbose=0)
        next_input = np.append(prediction_input[:, 1:, :], [[
            [pred[0][0]] * scaled_seq.shape[1]
        ]], axis=1)
        prediction_input = next_input
        predictions_scaled.append(pred[0][0])

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df['Close'].values.reshape(-1, 1)
    close_scaler.fit(close_prices)
    predictions = close_scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=future_days + 1, freq='B')[1:]
    return pd.DataFrame({'Date': future_dates, 'Predicted Future Price': predictions.flatten()})
