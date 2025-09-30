import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -------------------
# Helper Functions
# -------------------

# Prophet Prediction
def prophet_forecast(df, days=30):
    df_prophet = df[['Timestamp', 'CPU']].rename(columns={'Timestamp':'ds', 'CPU':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds','yhat']].tail(days).rename(columns={'ds':'Timestamp','yhat':'CPU'})

# LSTM Prediction
def lstm_forecast(df, days=30, lookback=3):
    data = df['CPU'].values
    X, y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    
    input_seq = data[-lookback:].reshape((1, lookback, 1))
    predictions = []
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:,1:,:], [[pred]], axis=1)
    future_dates = pd.date_range(df['Timestamp'].iloc[-1]+pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({'Timestamp': future_dates, 'CPU': predictions})

# Cost Calculation & Suggestions
def cost_suggestion(pred_df, cpu_price=0.05):
    pred_df['Estimated_Cost'] = pred_df['CPU'] * cpu_price
    pred_df['Action'] = np.where(pred_df['CPU']>70, 'Consider Scaling Up', 'Consider Scaling Down')
    return pred_df

# -------------------
# Streamlit UI
# -------------------
st.title("Cloud Usage & Cost Prediction Tool")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    # Read CSV and clean headers
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces

    # Map the column for CPU usage
    if 'cpu_util_percent' not in df.columns or 'date' not in df.columns:
        st.error("CSV must contain 'date' and 'cpu_util_percent' columns")
    else:
        df['Timestamp'] = pd.to_datetime(df['date'])
        df['CPU'] = df['cpu_util_percent']  # Rename for internal usage
        st.subheader("Uploaded Data")
        st.dataframe(df[['Timestamp','CPU']].tail())

        # Prediction settings
        days_to_predict = st.number_input("Days to Predict", min_value=7, max_value=90, value=30)
        model_choice = st.selectbox("Select Model", ["Prophet", "LSTM"])

        # Predict & Suggest
        if st.button("Predict & Suggest"):
            if model_choice=="Prophet":
                forecast = prophet_forecast(df, days=days_to_predict)
            else:
                forecast = lstm_forecast(df, days=days_to_predict)

            result_df = cost_suggestion(forecast)
            st.subheader("Predicted Usage & Cost")
            st.dataframe(result_df)

            st.subheader("Visualization")
            st.line_chart(result_df.set_index('Timestamp')['CPU'])
            st.line_chart(result_df.set_index('Timestamp')['Estimated_Cost'])

            st.subheader("Optimization Suggestions")
            st.dataframe(result_df[['Timestamp','Action','Estimated_Cost']])
