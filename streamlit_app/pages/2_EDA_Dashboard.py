"""
2_EDA_Dashboard.py — Exploratory Data Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.visualizations import (
    correlation_heatmap, feature_distribution_plot,
    missing_values_chart, class_distribution_chart,
)
from utils.data_loader import load_cached_dataset

from utils.theme import apply_custom_theme

st.set_page_config(page_title="🔍 EDA Dashboard", page_icon="🔍", layout="wide")

apply_custom_theme()

st.markdown("# 🔍 Exploratory Data Analysis")

if "dataset" not in st.session_state:
    # Fallback: try to load from dataset_path if available
    if "dataset_path" in st.session_state:
        df = load_cached_dataset(
            st.session_state["dataset_path"],
            st.session_state.get("dataset_type", "nsl-kdd")
        )
        st.session_state["dataset"] = df
    else:
        st.warning("⚠️ No dataset loaded. Go to **Dataset Explorer** first.")
        st.stop()

df = st.session_state["dataset"]

# ──────────────────────────────────────────────
# Missing Values
# ──────────────────────────────────────────────
st.markdown("## 🕳️ Missing Values Analysis")
fig_missing = missing_values_chart(df)
st.plotly_chart(fig_missing, use_container_width=True)

# ──────────────────────────────────────────────
# Class Distribution
# ──────────────────────────────────────────────
if "label" in df.columns:
    st.markdown("## 🎯 Class Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_dist = class_distribution_chart(df["label"])
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.markdown("### Breakdown")
        value_counts = df["label"].value_counts()
        for cls, count in value_counts.items():
            pct = count / len(df) * 100
            st.markdown(f"**{cls}**: {count:,} ({pct:.1f}%)")

# ──────────────────────────────────────────────
# Feature Distributions
# ──────────────────────────────────────────────
st.markdown("## 📊 Feature Distributions")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    selected_features = st.multiselect(
        "Select features to visualize",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
    )

    if selected_features:
        cols = st.columns(2)
        for i, feat in enumerate(selected_features):
            with cols[i % 2]:
                fig = feature_distribution_plot(df, feat)
                st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Correlation Heatmap
# ──────────────────────────────────────────────
st.markdown("## 🌡️ Correlation Heatmap")
with st.expander("Show Correlation Heatmap (may take a moment for large datasets)", expanded=False):
    fig_corr = correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

# ──────────────────────────────────────────────
# Data Types Summary
# ──────────────────────────────────────────────
st.markdown("## 📑 Data Types Summary")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
with col_b:
    st.metric("Categorical Features", len(df.select_dtypes(include=["object"]).columns))

dtype_summary = df.dtypes.value_counts().reset_index()
dtype_summary.columns = ["Data Type", "Count"]
st.dataframe(dtype_summary, use_container_width=True)
