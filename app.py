import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import plotly.graph_objs as go

class LSTMModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1O87HF4tdtM8Qyv7xa4HEoCUlZABY0jKc&export=download"
    return pd.read_csv(url)

@st.cache_resource
def load_xgb():
    return joblib.load("xgb_model.pkl")

@st.cache_resource
def load_lstm():
    try:
        model = LSTMModel()
        model.load_state_dict(torch.load("lstm_model.pt", map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        return None

st.title("Bitcoin Support–Resistance ML Trading Signals")

df = load_data()
xgb_model = load_xgb()
lstm_model = load_lstm()

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Support and Resistance Zones")
if "cluster" in df.columns and "level" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="BTC"))
    sr = df.groupby("cluster")["level"].mean().reset_index()
    for _, row in sr.iterrows():
        fig.add_hline(y=row["level"], line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Support/resistance columns missing")

st.subheader("XGBoost Prediction")
feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
latest = df.iloc[-1][feature_cols].values.reshape(1, -1)
xgb_pred = xgb_model.predict(latest)[0]
xgb_proba = xgb_model.predict_proba(latest)[0][1]
st.write("Signal:", "LONG" if xgb_pred == 1 else "SHORT")
st.write("Probability:", xgb_proba)

st.subheader("LSTM Prediction")
if lstm_model:
    seq = df.iloc[-30:][feature_cols]
    inp = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)
    lstm_output = torch.sigmoid(lstm_model(inp)).item()
    st.write("Signal:", "LONG" if lstm_output > 0.5 else "SHORT")
    st.write("Confidence:", lstm_output)
else:
    st.write("No LSTM model loaded")

st.subheader("Final Recommendation")
if xgb_pred == 1 and (lstm_model is None or lstm_output > 0.5):
    st.write("Both models bullish. LONG.")
elif xgb_pred == 0 and (lstm_model is None or lstm_output < 0.5):
    st.write("Both models bearish. SHORT.")
else:
    st.write("Mixed signals.")
