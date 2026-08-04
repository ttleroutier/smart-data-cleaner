import pandas as pd


def basic_diagnostics(df: pd.DataFrame) -> dict:
    """Return a simple diagnostic report for the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "empty_columns": int((df.isna().all()).sum()),
    }

