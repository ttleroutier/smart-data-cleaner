import pandas as pd


def load_file(uploaded_file):
    """Load a CSV or Excel file into a pandas DataFrame."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

