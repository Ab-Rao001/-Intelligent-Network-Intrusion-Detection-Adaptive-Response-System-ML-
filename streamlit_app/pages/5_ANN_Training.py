"""
5_ANN_Training.py — Neural Network Training Module
"""

import streamlit as st
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

from classification.ann_model import ANNClassifier, SVMClassifier, ATTACK_CLASSES
from utils.visualizations import training_history_plot
from utils.metrics import compute_classification_metrics
from utils.data_loader import get_cached_preprocessing

from utils.theme import apply_custom_theme

st.set_page_config(page_title="🧠 ANN Training", page_icon="🧠", layout="wide")

apply_custom_theme()

st.markdown("# 🧠 Model Training")
st.markdown("Train an Artificial Neural Network (ANN) or SVM for attack classification.")

if not st.session_state.get("pca_ran", False):
    st.warning("⚠️ Run PCA first (which also preprocesses data) on the **PCA Visualization** page.")
    st.stop()

dataset_path = st.session_state["dataset_path"]
dataset_type = st.session_state.get("dataset_type", "nsl-kdd")
X_train, X_test, y_train, y_test, feats, _ = get_cached_preprocessing(dataset_path, dataset_type)

# Ensure downstream pages (Prediction, Metrics) can access the data
st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test
st.session_state["feature_names"] = feats

st.info(f"📦 Dataset ready — Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples, Features: {X_train.shape[1]}")

# ──────────────────────────────────────────────
# Model Selection & Configuration
# ──────────────────────────────────────────────
tab_ann, tab_svm = st.tabs(["🧠 ANN (Deep Learning)", "📊 SVM (Optional)"])

with tab_ann:
    st.markdown("### ANN Configuration")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Epochs", 5, 100, 30)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
        )

    with col2:
        layer1 = st.slider("Hidden Layer 1 (neurons)", 32, 256, 128)
        layer2 = st.slider("Hidden Layer 2 (neurons)", 16, 128, 64)
        layer3 = st.slider("Hidden Layer 3 (neurons)", 8, 64, 32)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, step=0.05)

    if st.button("🚀 Train ANN", type="primary"):
        st.session_state["ann_training_active"] = True
        st.session_state["ann_training_done"] = False
        st.session_state["ann_progress"] = {"epoch": 0, "logs": {}}
        st.rerun()

    if st.session_state.get("ann_training_active", False):
        st.info("🔄 ANN Training running in background...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def train_task():
            try:
                ann = ANNClassifier(
                    input_dim=X_train.shape[1],
                    hidden_layers=[layer1, layer2, layer3],
                    dropout_rate=dropout,
                    learning_rate=learning_rate,
                )

                def progress_cb(epoch, logs):
                    st.session_state["ann_progress"] = {"epoch": epoch + 1, "logs": logs}

                history = ann.train(
                    X_train, y_train, X_test, y_test,
                    epochs=epochs, batch_size=batch_size,
                    save_dir=os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd")),
                    progress_callback=progress_cb
                )

                results = ann.evaluate(X_test, y_test)

                st.session_state["ann_model"] = ann
                st.session_state["ann_history"] = history.history
                st.session_state["ann_results"] = results
                # Clear any previous errors
                st.session_state.pop("ann_training_error", None)
            except Exception as e:
                st.session_state["ann_training_error"] = str(e)
            finally:
                st.session_state["ann_training_active"] = False
                st.session_state["ann_training_done"] = True

        # Start thread if not already running
        if "ann_thread" not in st.session_state or not st.session_state["ann_thread"].is_alive():
            thread = threading.Thread(target=train_task)
            add_script_run_ctx(thread)
            thread.start()
            st.session_state["ann_thread"] = thread

        # Polling loop
        while st.session_state.get("ann_training_active", True):
            prog = st.session_state.get("ann_progress", {})
            current_epoch = prog.get("epoch", 0)
            logs = prog.get("logs", {})
            
            pct = min(current_epoch / epochs, 1.0)
            progress_bar.progress(pct)
            status_text.text(f"Epoch {current_epoch}/{epochs} | Loss: {logs.get('loss', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
            time.sleep(1)
            st.rerun()

    if st.session_state.get("ann_training_done", False):
        if "ann_training_error" in st.session_state:
            st.error(f"❌ Training Failed: {st.session_state['ann_training_error']}")
        else:
            results = st.session_state.get("ann_results", {})
            if results:
                st.success(f"✅ ANN trained! Test Accuracy: **{results.get('accuracy', 0):.4f}**")

    # Show results
    if "ann_history" in st.session_state:
        st.markdown("### 📈 Training Curves")
        fig_hist = training_history_plot(st.session_state["ann_history"])
        st.plotly_chart(fig_hist, use_container_width=True)

        results = st.session_state["ann_results"]
        st.markdown("### 📊 Test Results")
        c1, c2, c3, c4 = st.columns(4)
        metrics = compute_classification_metrics(y_test, results["y_pred"])
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1 Score", f"{metrics['f1']:.4f}")

        # Model summary
        with st.expander("📋 Model Architecture"):
            ann = st.session_state["ann_model"]
            st.code(ann.summary(), language="text")

with tab_svm:
    st.markdown("### SVM Configuration")
    st.warning("⚠️ SVM can be very slow on large datasets. Consider using a subset.")

    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
    c_val = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
    use_subset = st.checkbox("Use subset (first 5000 samples)", value=True)

    if st.button("🚀 Train SVM", type="primary"):
        with st.spinner("Training SVM..."):
            if use_subset:
                X_tr = X_train[:5000]
                y_tr = y_train[:5000]
            else:
                X_tr = X_train
                y_tr = y_train

            svm = SVMClassifier(kernel=kernel, C=c_val)
            svm.train(X_tr, y_tr)
            results = svm.evaluate(X_test, y_test)

            st.session_state["svm_model"] = svm
            st.session_state["svm_results"] = results

        st.success(f"✅ SVM trained! Test Accuracy: **{results['accuracy']:.4f}**")

    if "svm_results" in st.session_state:
        results = st.session_state["svm_results"]
        st.markdown("### 📊 SVM Test Results")
        c1, c2, c3, c4 = st.columns(4)
        metrics = compute_classification_metrics(y_test, results["y_pred"])
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1 Score", f"{metrics['f1']:.4f}")
