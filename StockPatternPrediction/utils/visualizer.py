import plotly.graph_objs as go

def plot_price_and_prediction(hist_df, pred_df):
    required_cols = ["Open", "High", "Low", "Close", "Date"]
    if hist_df.empty or any(col not in hist_df.columns for col in required_cols):
        fig = go.Figure()
        fig.update_layout(title="No historical data available")
        return fig

    if hist_df[["Open", "High", "Low", "Close"]].isnull().any().any():
        fig = go.Figure()
        fig.update_layout(title="Missing values in historical data")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_df['Date'],
        open=hist_df['Open'], high=hist_df['High'],
        low=hist_df['Low'], close=hist_df['Close'],
        name='Historical Price'
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['Date'], y=pred_df['Predicted Close Price'],
        mode='lines+markers', name='Predicted Trend',
        line=dict(dash='dash', color='cornflowerblue')
    ))

    fig.update_layout(title="Stock Price - Historical and Predicted")
    return fig


# NEW FUTURE FORECAST VISUAL
def plot_future_predictions(past_df, future_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=past_df['Date'], y=past_df['Predicted Close Price'],
        mode='lines', name='Past Predictions', line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=future_df['Date'], y=future_df['Predicted Future Price'],
        mode='lines+markers', name='Future Forecast',
        line=dict(dash='dot', color='orange')
    ))

    fig.update_layout(
        title="Past vs Future Predictions",
        xaxis_title="Date", yaxis_title="Price"
    )

    return fig
