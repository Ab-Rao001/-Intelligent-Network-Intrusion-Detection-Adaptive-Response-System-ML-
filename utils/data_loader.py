"""
utils/data_loader.py — Cached data loading functions
"""

import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocessor import DataPreprocessor

@st.cache_data(show_spinner=False)
def load_cached_dataset(filepath: str, dataset_type: str = "nsl-kdd") -> pd.DataFrame:
    """Loads a dataset into memory, cached by filepath."""
    if dataset_type == "nsl-kdd":
        from preprocessing.preprocessor import NSL_KDD_COLUMNS
        try:
            df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
            if "difficulty_level" in df.columns:
                df.drop(columns=["difficulty_level"], inplace=True)
        except Exception:
            df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    return df

@st.cache_data(show_spinner=False)
def get_cached_preprocessing(filepath: str, dataset_type: str, n_samples: int = None):
    """
    Runs the preprocessing pipeline and caches the resulting heavy numpy arrays.
    Returns: X_train, X_test, y_train, y_test, feats, df_clean
    """
    pp = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats, df_clean = pp.run_pipeline(
        filepath, dataset_type=dataset_type, n_samples=n_samples
    )
    return X_train, X_test, y_train, y_test, feats, df_clean

@st.cache_data(show_spinner=False)
def get_cached_pca(X_train, n_components: int):
    """
    Runs PCA clustering and caches the transformed data to save memory.
    """
    from clustering.cluster_model import ClusteringPipeline
    pipeline = ClusteringPipeline(n_pca_components=n_components)
    pca_data = pipeline.fit_pca(X_train)
    variance = pipeline.get_explained_variance()
    return pca_data, variance, pipeline
