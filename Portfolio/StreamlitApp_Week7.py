import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap


# ── Setup & Path Configuration ────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# ── Secrets ───────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────────
# Partner ticker  = valid_partner identified in the notebook (e.g., 'AME')
# Target ticker   = target_ticker identified in the notebook (e.g., 'TXN')
# Update PARTNER_TICKER and TARGET_TICKER to match your notebook's output.
PARTNER_TICKER = 'EXPD'  # valid_partner from HW4 notebook
TARGET_TICKER  = 'AOS'   # target_ticker from HW4 notebook

df_features = extract_features_pair(
    partner_ticker=PARTNER_TICKER,
    target_ticker=TARGET_TICKER
)

MODEL_INFO = {
    "endpoint":  aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline":  "finalized_pair_model.tar.gz",
    # Column order must match training: [partner, target]
    "keys":   [PARTNER_TICKER, TARGET_TICKER],
    "inputs": [
        {"name": PARTNER_TICKER, "label": f"Partner Stock Price ({PARTNER_TICKER})",
         "type": "number", "min": 0.01, "default": 100.0, "step": 0.01},
        {"name": TARGET_TICKER,  "label": f"Target Stock Price ({TARGET_TICKER})",
         "type": "number", "min": 0.01, "default": 150.0, "step": 0.01},
    ]
}

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, s3_key_prefix):
    """Download and load the sklearn pipeline from S3."""
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    local_tar = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{s3_key_prefix}/{os.path.basename(filename)}",
        Filename=local_tar
    )

    extract_dir = tempfile.mkdtemp()
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(os.path.join(extract_dir, joblib_file))


@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_path):
    """Download and load the SHAP explainer from S3."""
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=s3_key,
            Filename=local_path
        )

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    """Send input_df to the SageMaker endpoint and return (prediction, status)."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values)
        # The endpoint returns the signal for the last row (the new input)
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        return pred_val, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── SHAP Explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df, _session, bucket):
    """Load the SHAP explainer and render a waterfall plot for the latest row."""
    explainer_name = MODEL_INFO["explainer"]
    local_explainer = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer = load_shap_explainer(
        _session,
        bucket,
        posixpath.join('explainer', explainer_name),
        local_explainer
    )

    best_pipeline = load_pipeline(_session, bucket, 'sklearn-pipeline-deployment')

    # Preprocessing = all steps except the last two (sampler + model)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    X_transformed = preprocessing_pipeline.transform(input_df)

    # Get feature names after SelectKBest
    try:
        feat_names = best_pipeline.named_steps['feature_selection'].get_feature_names_out()
    except Exception:
        feat_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

    X_transformed_df = pd.DataFrame(X_transformed, columns=feat_names)

    # Compute SHAP values — use the last row (the new user input)
    shap_values = explainer(X_transformed_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[-1, :, 0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # Most influential feature
    try:
        top_feature = (
            pd.Series(
                shap_values[-1, :, 0].values,
                index=shap_values[-1, :, 0].feature_names
            )
            .abs()
            .idxmax()
        )
        st.info(
            f"**Business Insight:** The most influential factor in this "
            f"decision was **{top_feature}**."
        )
    except Exception:
        pass


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pairs Trading Signal", layout="wide")
st.title("📈 Pairs Trading Signal Predictor")
st.markdown(
    """
    Enter the current prices for the cointegrated stock pair to predict the trading signal.
    - **1** = 🟢 BUY  
    - **0** = 🟡 HOLD  
    - **-1** = 🔴 SELL
    """
)

with st.form("pred_form"):
    st.subheader("Input Prices")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=float(inp["min"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )

    submitted = st.form_submit_button("Predict Signal")

if submitted:
    # Build input row in training column order
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Append the new row to the historical data so the custom
    # PairFeatureEngineer can compute rolling features correctly
    base_df  = df_features.copy()
    input_df = pd.concat(
        [base_df, pd.DataFrame([data_row], columns=base_df.columns)],
        ignore_index=True
    )

    pred, status = call_model_api(input_df)

    if status == 200:
        signal_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}
        label = signal_map.get(pred, str(pred))
        st.metric("Predicted Trading Signal", label)
        st.write(f"Raw prediction value: **{pred}**")

        with st.spinner("Generating SHAP explanation..."):
            display_explanation(input_df, session, aws_bucket)
    else:
        st.error(pred)
