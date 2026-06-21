"""
Dataset loading utilities.
"""

import pandas as pd


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file uploaded through Streamlit.

    Parameters
    ----------
    uploaded_file
        Streamlit uploaded file object.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    return pd.read_csv(uploaded_file)