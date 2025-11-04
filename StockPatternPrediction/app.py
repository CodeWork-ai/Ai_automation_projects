import streamlit as st
from utils.data_loader import fetch_stock_data
from utils.lstm_predictor import train_lstm_predictor, predict_future
from utils.visualizer import plot_price_and_prediction, plot_future_predictions
from config.tickers import TOP_30_TICKERS

st.title("AI Stock Pattern Prediction with LSTM")

ticker = st.selectbox("Select Company Ticker", TOP_30_TICKERS)
days_ahead = st.slider("Days to Predict Ahead", 5, 30, 7)

hist_data = fetch_stock_data(ticker)
st.write("Historical Data Preview:", hist_data.head())

with st.spinner("Training LSTM model..."):
    model, scaler, metrics, predicted = train_lstm_predictor(hist_data, seq_length=50)

if metrics is not None:
    st.subheader("LSTM Model Performance Metrics")
    st.markdown(f"* Mean Absolute Error (MAE): `{metrics['MAE']:.3f}`")
    st.markdown(f"* Mean Squared Error (MSE): `{metrics['MSE']:.3f}`")
    st.markdown(f"* R² Score: `{metrics['R2 Score']:.3f}`")
    st.markdown(f"* Approx Accuracy: `{metrics['Approx Accuracy (%)']:.2f}%`")

    st.subheader(f"{ticker} Price Action and Prediction")
    fig = plot_price_and_prediction(hist_data, predicted)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted Price Data")
    st.dataframe(predicted)

    if st.button("Predict Next Few Days"):
        st.info(f"Generating next {days_ahead} days forecast...")
        future_df = predict_future(model, hist_data, scaler, seq_length=50, future_days=days_ahead)
        st.dataframe(future_df)

        future_fig = plot_future_predictions(predicted, future_df)
        st.plotly_chart(future_fig, use_container_width=True)
