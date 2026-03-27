import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

from sklearn.pipeline import Pipeline
import shap


# ── Setup & Path Configuration ─────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# ── Secrets ────────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]       # gabe-wilson-s3-bucket
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]     # logistic-pipeline-endpoint-auto-6

# ── AWS Session ────────────────────────────────────────────────────────────────
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

# ── Load model metadata (column names saved by the notebook) ──────────────────
@st.cache_resource
def load_metadata(_session, bucket):
    """
    Download model_metadata.json from S3 and return the input column names.
    The notebook saves: {"input_cols": [valid_partner_ticker, "TXN"]}
    """
    s3_client = _session.client('s3')
    local_path = os.path.join(tempfile.gettempdir(), 'model_metadata.json')
    s3_client.download_file(
        Bucket=bucket,
        Key='sklearn-pipeline-deployment/model_metadata.json',
        Filename=local_path
    )
    with open(local_path) as f:
        return json.load(f)

metadata   = load_metadata(session, aws_bucket)
input_cols = metadata['input_cols']   # e.g. ['NVDA', 'TXN']

# ── Data & Model Configuration ────────────────────────────────────────────────
# Fetch the last year of price history for the pair; rename columns to match
# whatever the pipeline was actually fitted on (from model_metadata.json)
df_features = extract_features_pair()
df_features.columns = input_cols      # align placeholder names to real ticker names

MODEL_INFO = {
    "endpoint": aws_endpoint,                          # logistic-pipeline-endpoint-auto-6
    "explainer": 'explainer_pair.shap',
    "pipeline":  'finalized_pair_model.tar.gz',
    "s3_pipeline_key": 'sklearn-pipeline-deployment', # S3 prefix for the tar.gz
    "keys":   input_cols,                              # [valid_partner, 'TXN']
    "inputs": [
        {"name": col, "type": "number", "min": 0.0, "default": 0.0, "step": 10.0}
        for col in input_cols
    ]
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, s3_key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    local_tar = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{s3_key}/{os.path.basename(filename)}",
        Filename=local_tar
    )

    extract_dir = tempfile.gettempdir()
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(os.path.join(extract_dir, joblib_file))


@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    """
    Send the last row of input_df to the SageMaker endpoint as CSV.
    inference_pair.py expects: <col0_price>,<col1_price>
    Returns (prediction_dict, status_code).
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),    # inference_pair.py input_fn handles text/csv
        deserializer=JSONDeserializer()
    )

    try:
        # Send only the new input row (last row appended by the app)
        row_csv = ','.join(str(v) for v in input_df.iloc[-1].values)
        result  = predictor.predict(row_csv)
        # result = {"prediction": int, "probabilities": [...]}
        return result, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── SHAP Explainability ───────────────────────────────────────────────────────
def display_explanation(input_df, _session, bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_explainer = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer     = load_shap_explainer(
        _session, bucket,
        posixpath.join('explainer', explainer_name),
        local_explainer
    )
    best_pipeline = load_pipeline(_session, bucket, MODEL_INFO["s3_pipeline_key"])

    # Re-build the preprocessing sub-pipeline (everything except the final estimator)
    # Steps: pair_ind_5 → imputer → scaler → feature_selection  (indices 0-3, i.e. [:-1])
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_transformed       = preprocessing_pipeline.transform(input_df)

    # Get feature names after SelectKBest (step index 3 = feature_selection)
    feature_names           = best_pipeline[:-1].get_feature_names_out()
    input_transformed_df    = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_transformed_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 0].values, index=shap_values[0, :, 0].feature_names)
        .abs()
        .idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TXN Pairs Trading — ML Deployment", layout="wide")
st.title("👨‍💻 TXN Pairs Trading — ML Deployment")
st.caption(f"Pair: **{input_cols[0]}** (partner) vs **{input_cols[1]}** (target: TXN)")

with st.form("pred_form"):
    st.subheader("Enter current stock prices")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].upper(),
                min_value=inp['min'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Append the new price row to the historical data so PairFeatureEngineer
    # has enough context (window=60) to compute spread/zscore features
    new_row  = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.concat(
        [df_features, pd.DataFrame([new_row], columns=df_features.columns)],
        ignore_index=True
    )

    result, status = call_model_api(input_df)

    if status == 200:
        signal_map = {1: "BUY 📈", 0: "HOLD ⏸️", -1: "SELL 📉"}
        pred_int   = result.get("prediction", 0)
        pred_label = signal_map.get(pred_int, str(pred_int))
        probas     = result.get("probabilities", [])

        st.metric("Prediction Signal", pred_label)

        if probas:
            prob_df = pd.DataFrame(
                {"Signal": list(signal_map.values()), "Probability": probas}
            )
            st.bar_chart(prob_df.set_index("Signal"))

        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(result)
