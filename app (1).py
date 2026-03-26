import os
import sys
import warnings
import tempfile
import tarfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import pickle

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

# ── Path & warning setup ──────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair, FEATURE_COLS, TARGET, PAIR_STOCK

# ── AWS secrets ───────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint":   aws_endpoint,
    "explainer":  "explainer_pair.shap",
    "pipeline":   "finalized_pair_model.tar.gz",
    # Keys / inputs now match FEATURE_COLS from the TXN notebook
    "keys":   FEATURE_COLS,
    "inputs": [
        {"name": "Zscore",        "default":  0.0,   "step": 0.01,  "min": -10.0},
        {"name": "Zscore_Lag1",   "default":  0.0,   "step": 0.01,  "min": -10.0},
        {"name": "Zscore_MA5",    "default":  0.0,   "step": 0.01,  "min": -10.0},
        {"name": "Spread",        "default": 10.0,   "step": 0.10,  "min": -500.0},
        {"name": "Spread_Lag1",   "default": 10.0,   "step": 0.10,  "min": -500.0},
        {"name": "Spread_Lag2",   "default": 10.0,   "step": 0.10,  "min": -500.0},
        {"name": "Spread_Change", "default":  0.0,   "step": 0.01,  "min": -100.0},
        {"name": "Return_TXN",    "default":  0.0,   "step": 0.001, "min": -1.0},
        {"name": "Return_Pair",   "default":  0.0,   "step": 0.001, "min": -1.0},
        {"name": "Vol_Ratio",     "default":  1.0,   "step": 0.01,  "min":  0.0},
    ],
}

SIGNAL_MAP   = {0: "🔴  SELL", 1: "🟡  HOLD", 2: "🟢  BUY"}
SIGNAL_COLOR = {0: "red",      1: "orange",    2: "green"}

# ── Data loader (cached) ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    """Download the last 365 days of TXN / pair prices and compute features."""
    return extract_features_pair()

df_features = load_features()

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket: str, s3_key: str):
    """Download finalized_pair_model.tar.gz from S3 and return the pipeline."""
    s3_client = _session.client("s3")
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{s3_key}/{os.path.basename(filename)}",
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        # Support both .joblib and .pkl inside the archive
        members = tar.getnames()
        pkl_file = next(
            (f for f in members if f.endswith(".pkl") or f.endswith(".joblib")),
            None,
        )

    if pkl_file is None:
        raise FileNotFoundError("No .pkl / .joblib found inside the model archive.")

    return joblib.load(pkl_file)


@st.cache_resource
def load_shap_explainer(_session, bucket: str, s3_key: str, local_path: str):
    """Download explainer_pair.shap from S3 (skip if already local) and load it."""
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=s3_key,
        )

    with open(local_path, "rb") as f:
        return pickle.load(f)          # saved with pickle.dump in the notebook


# ── Prediction logic ──────────────────────────────────────────────────────────
def call_model_api(input_array: np.ndarray):
    """
    Send *input_array* (shape 1 × 10) to the SageMaker endpoint.
    Returns (prediction_int, 200) on success or (error_str, 500) on failure.
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred = predictor.predict(input_array)
        pred_val = int(np.array(raw_pred).flatten()[0])
        return pred_val, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── SHAP explanation panel ────────────────────────────────────────────────────
def display_explanation(input_df: pd.DataFrame, _session, bucket: str):
    """
    Transform *input_df* through the pipeline's preprocessing steps,
    compute SHAP values, and render a waterfall plot + business insight.
    """
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)
    s3_explainer_key = posixpath.join("explainer", explainer_name)

    explainer     = load_shap_explainer(_session, bucket, s3_explainer_key, local_path)
    best_pipeline = load_pipeline(_session, bucket, "sklearn-pipeline-deployment")

    # Steps 0 (imputer) + 1 (scaler) — skip SMOTE (index 2) and LR (index 3)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:2])
    input_transformed      = preprocessing_pipeline.transform(input_df)
    input_transformed_df   = pd.DataFrame(input_transformed, columns=FEATURE_COLS)

    shap_values = explainer(input_transformed_df)

    # waterfall for the first (and only) row; class 0 slice for multiclass
    shap_obj = shap_values[0] if shap_values.values.ndim == 2 else shap_values[0, :, 0]

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_obj, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # Most influential feature
    top_feature = (
        pd.Series(np.abs(shap_obj.values), index=shap_obj.feature_names)
        .idxmax()
    )
    st.info(
        f"**Business Insight:** The most influential factor in this decision "
        f"was **{top_feature}**."
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TXN Pair Trading Signal", page_icon="📈", layout="wide")

st.title("📈 TXN Pair Trading Signal Predictor")
st.markdown(
    f"Enter the current spread features for **{TARGET} / {PAIR_STOCK}** "
    f"to receive a **BUY / HOLD / SELL** prediction from the deployed model."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    f"""
    **Target ticker:** `{TARGET}` (Texas Instruments)  
    **Pair ticker:**   `{PAIR_STOCK}` (Agilent Technologies)  

    **Model:** Logistic Regression with:
    - ElasticNet regularization (Lasso + Ridge)
    - SMOTE oversampling (class imbalance)
    - GridSearchCV tuning (`C`, `l1_ratio`, `k_neighbors`)

    **Signals**
    | Code | Meaning |
    |------|---------|
    | 🟢 BUY  | Z-score < −1 → spread likely to revert up |
    | 🔴 SELL | Z-score >  1 → spread likely to revert down |
    | 🟡 HOLD | Z-score within ±1 → no strong signal |
    """
)

# ── Live feature snapshot ─────────────────────────────────────────────────────
with st.expander("📊 Latest live feature snapshot (last row)", expanded=False):
    if df_features is not None and not df_features.empty:
        latest = df_features.iloc[[-1]].T.rename(columns={df_features.index[-1]: "Value"})
        st.dataframe(latest.style.format("{:.5f}"))
    else:
        st.warning("Could not load live features. Check yfinance connectivity.")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("pred_form"):
    st.subheader("🔢 Input Features")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                label=inp["name"].replace("_", " "),
                min_value=float(inp["min"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
                format="%.5f",
            )

    submitted = st.form_submit_button("🔮 Run Prediction", type="primary")

# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    # Build a single-row DataFrame that matches FEATURE_COLS order
    input_df    = pd.DataFrame([[user_inputs[k] for k in FEATURE_COLS]], columns=FEATURE_COLS)
    input_array = input_df.values  # shape (1, 10)

    with st.spinner("Calling SageMaker endpoint…"):
        result, status = call_model_api(input_array)

    st.markdown("---")
    if status == 200:
        signal_text  = SIGNAL_MAP.get(result, f"Unknown ({result})")
        signal_color = SIGNAL_COLOR.get(result, "gray")

        st.subheader("📊 Prediction Result")
        st.markdown(
            f"<h2 style='color:{signal_color};text-align:center'>{signal_text}</h2>",
            unsafe_allow_html=True,
        )

        # Show SHAP explanation
        try:
            display_explanation(input_df, session, aws_bucket)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.error(result)
