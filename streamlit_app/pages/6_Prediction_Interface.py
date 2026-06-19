"""
6_Prediction_Interface.py — Real-time Attack Prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.ann_model import ATTACK_CLASSES

from utils.theme import apply_custom_theme

st.set_page_config(page_title="⚡ Prediction", page_icon="⚡", layout="wide")

apply_custom_theme()

st.markdown("# ⚡ Attack Prediction Interface")
st.markdown("Predict the attack type for network traffic in real-time.")

if "ann_model" not in st.session_state:
    st.warning("⚠️ Train an ANN model first on the **ANN Training** page.")
    st.stop()

ann = st.session_state["ann_model"]
feature_names = st.session_state.get("feature_names", [])
preprocessor = st.session_state.get("preprocessor", None)

# ──────────────────────────────────────────────
# Prediction Methods
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎲 Random Sample", "✏️ Manual Input", "📁 Batch Upload"])

with tab1:
    st.markdown("### Predict on a random test sample")

    # Ensure X_test and y_test are available (populate from preprocessing cache if needed)
    if "X_test" not in st.session_state and "dataset_path" in st.session_state:
        from utils.data_loader import get_cached_preprocessing
        dataset_path = st.session_state["dataset_path"]
        dataset_type = st.session_state.get("dataset_type", "nsl-kdd")
        X_train, X_test, y_train, y_test, feats, _ = get_cached_preprocessing(dataset_path, dataset_type)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = feats

    if "X_test" in st.session_state:
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        if st.button("🎲 Pick Random Sample", type="primary"):
            idx = np.random.randint(0, len(X_test))
            sample = X_test[idx:idx+1]
            true_label = ATTACK_CLASSES[y_test[idx]]

            # Predict
            pred_idx = ann.predict(sample)[0]
            pred_label = ATTACK_CLASSES[pred_idx]
            probs = ann.predict_proba(sample)[0]

            # Display
            is_normal = pred_label == "Normal"
            css_class = "pred-normal" if is_normal else "pred-attack"
            icon = "✅" if is_normal else "🚨"

            st.markdown(f"""
            <div class="prediction-result {css_class}">
                <div style="font-size: 3rem;">{icon}</div>
                <div style="font-size: 2rem; font-weight: 700; color: {'#00d4aa' if is_normal else '#ff4757'};">
                    {pred_label}
                </div>
                <div style="color: #8892b0; margin-top: 8px;">
                    True Label: <b>{true_label}</b> | 
                    Correct: <b>{'✅ Yes' if pred_label == true_label else '❌ No'}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("#### Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": ATTACK_CLASSES,
                "Probability": probs,
            })
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x:.4f}")
            st.dataframe(prob_df, use_container_width=True)

with tab2:
    st.markdown("### Enter feature values manually")
    st.info("Adjust the sliders to set feature values (scaled 0–1 after preprocessing).")

    n_features = st.session_state.get("X_train", np.zeros((1, 41))).shape[1]
    input_values = []

    cols = st.columns(4)
    for i in range(min(n_features, 20)):  # show first 20 features
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        with cols[i % 4]:
            val = st.number_input(fname, 0.0, 1.0, 0.0, step=0.01, key=f"feat_{i}")
            input_values.append(val)

    # Pad remaining features with 0.0 (prevent corrupting one-hot encodings)
    while len(input_values) < n_features:
        input_values.append(0.0)

    if st.button("⚡ Predict", type="primary", key="manual_predict"):
        sample = np.array(input_values).reshape(1, -1).astype(np.float32)
        pred_idx = ann.predict(sample)[0]
        pred_label = ATTACK_CLASSES[pred_idx]
        probs = ann.predict_proba(sample)[0]

        is_normal = pred_label == "Normal"
        css_class = "pred-normal" if is_normal else "pred-attack"
        icon = "✅" if is_normal else "🚨"

        st.markdown(f"""
        <div class="prediction-result {css_class}">
            <div style="font-size: 3rem;">{icon}</div>
            <div style="font-size: 2rem; font-weight: 700; color: {'#00d4aa' if is_normal else '#ff4757'};">
                {pred_label}
            </div>
            <div style="color: #8892b0; margin-top: 8px;">
                Confidence: <b>{probs[pred_idx]*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Upload a CSV for batch prediction")
    uploaded = st.file_uploader("Upload preprocessed CSV (numeric features only)", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} samples")

        if st.button("⚡ Batch Predict", type="primary"):
            X_batch = batch_df.values.astype(np.float32)
            preds = ann.predict(X_batch)
            pred_labels = [ATTACK_CLASSES[p] for p in preds]

            batch_df["Predicted_Class"] = pred_labels
            st.dataframe(batch_df, use_container_width=True)

            # Summary
            st.markdown("#### Prediction Summary")
            summary = pd.Series(pred_labels).value_counts()
            for cls, count in summary.items():
                st.write(f"**{cls}**: {count} ({count/len(pred_labels)*100:.1f}%)")
