import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
from io import StringIO

# ─── Configuration ────────────────────────────────────────────────────────────
ENDPOINT_NAME = "logistic-pipeline-endpoint-auto-6"
REGION = "us-east-1"

st.set_page_config(page_title="Pairs Trading Signal", layout="centered")
st.title("📈 Pairs Trading Signal Predictor")
st.markdown("""
Enter the current prices for the stock pair to predict the trading signal.
- **1** = BUY
- **0** = HOLD
- **-1** = SELL
""")

# ─── Input Section ────────────────────────────────────────────────────────────
st.header("Input Prices")

partner_price = st.number_input(
    "Partner Stock Price (e.g., AME)",
    min_value=0.01, value=100.0, step=0.01
)
target_price = st.number_input(
    "Target Stock Price (e.g., AAPL)",
    min_value=0.01, value=150.0, step=0.01
)

# ─── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict Signal"):
    try:
        # Prepare input as CSV
        input_df = pd.DataFrame({
            "partner": [partner_price],
            "target":  [target_price]
        })
        csv_buffer = StringIO()
        input_df.to_csv(csv_buffer, header=False, index=False)
        payload = csv_buffer.getvalue()

        # Call SageMaker endpoint
        client = boto3.client("sagemaker-runtime", region_name=REGION)
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )
        result = json.loads(response["Body"].read().decode())
        signal = result[0] if isinstance(result, list) else result

        signal_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}
        st.success(f"Predicted Signal: **{signal_map.get(signal, signal)}**")

    except Exception as e:
        st.error(f"Error calling endpoint: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Pairs Trading ML App | AWS SageMaker + Streamlit")
