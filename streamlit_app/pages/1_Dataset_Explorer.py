"""
1_Dataset_Explorer.py — Upload and explore network intrusion datasets
"""

import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.visualizations import class_distribution_chart
from utils.data_loader import load_cached_dataset

from utils.theme import apply_custom_theme

st.set_page_config(page_title="📊 Dataset Explorer", page_icon="📊", layout="wide")

apply_custom_theme()

st.markdown("# 📊 Dataset Explorer")
st.markdown("Upload your NSL-KDD or CICIDS2017 dataset, or generate a synthetic demo dataset.")

# ──────────────────────────────────────────────
# Upload / Generate
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Dataset", "🔧 Generate Synthetic"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"],
        help="Upload NSL-KDD (headerless) or CICIDS2017 (with headers)"
    )
    dataset_type = st.selectbox("Dataset Type", ["nsl-kdd", "cicids"])

    if uploaded_file:
        # Save temporarily
        temp_dir = os.path.join(PROJECT_ROOT, "dataset", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_upload.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["dataset_path"] = temp_path
        st.session_state["dataset_type"] = dataset_type
        
        df = load_cached_dataset(temp_path, dataset_type)
        st.session_state["dataset"] = df
        st.success(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

with tab2:
    st.markdown("Generate a realistic synthetic NSL-KDD dataset for demonstration.")
    n_samples = st.slider("Number of samples", 5000, 100000, 25000, step=5000)
    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("🔧 Generate Dataset", type="primary"):
        with st.spinner("Generating synthetic dataset..."):
            from dataset.generate_synthetic import generate_synthetic_dataset
            save_path = os.path.join(PROJECT_ROOT, "dataset", "synthetic_nsl_kdd.csv")
            # Only generate if it doesn't exist to save time/memory, or overwrite. Let's overwrite.
            generate_synthetic_dataset(n_samples=n_samples, output_path=save_path, seed=seed)
            st.session_state["dataset_path"] = save_path
            st.session_state["dataset_type"] = "nsl-kdd"
            df = load_cached_dataset(save_path, "nsl-kdd")
            st.session_state["dataset"] = df
            st.success(f"✅ Generated {len(df):,} samples → `dataset/synthetic_nsl_kdd.csv`")

# ──────────────────────────────────────────────
# Display Dataset Info
# ──────────────────────────────────────────────
if "dataset_path" in st.session_state:
    df = load_cached_dataset(
        st.session_state["dataset_path"], 
        st.session_state.get("dataset_type", "nsl-kdd")
    )
    st.session_state["dataset"] = df

    st.markdown("---")
    st.markdown("## 📋 Dataset Overview")

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-box">
            <div style="color:#00d4aa;font-size:1.8rem;font-weight:700">{df.shape[0]:,}</div>
            <div style="color:#8892b0">Rows</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <div style="color:#ffa502;font-size:1.8rem;font-weight:700">{df.shape[1]}</div>
            <div style="color:#8892b0">Features</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        missing = df.isnull().sum().sum()
        st.markdown(f"""<div class="stat-box">
            <div style="color:#ff4757;font-size:1.8rem;font-weight:700">{missing:,}</div>
            <div style="color:#8892b0">Missing Values</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        n_classes = df["label"].nunique() if "label" in df.columns else "N/A"
        st.markdown(f"""<div class="stat-box">
            <div style="color:#7c4dff;font-size:1.8rem;font-weight:700">{n_classes}</div>
            <div style="color:#8892b0">Unique Labels</div>
        </div>""", unsafe_allow_html=True)

    # Data preview
    st.markdown("### 🔎 Data Preview")
    st.dataframe(df.head(20), use_container_width=True, height=400)

    # Statistics
    st.markdown("### 📊 Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # Class distribution
    if "label" in df.columns:
        st.markdown("### 🎯 Class Distribution")
        fig = class_distribution_chart(df["label"])
        st.plotly_chart(fig, use_container_width=True)

    # Column info
    with st.expander("📑 Column Types & Info"):
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null": df.notnull().sum(),
            "Missing": df.isnull().sum(),
            "Unique": df.nunique(),
        }).reset_index(drop=True)
        st.dataframe(col_info, use_container_width=True)
else:
    st.info("👆 Upload a dataset or generate a synthetic one to get started.")
