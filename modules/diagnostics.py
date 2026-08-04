"""
diagnostics.py
Analysis functions for the Smart Data Cleaner app.

Rules:
- These functions NEVER modify the DataFrame.
- They only return information: statistics, previews, warnings.
"""

import pandas as pd
import numpy as np


# -----------------------------
# GLOBAL DIAGNOSTICS
# -----------------------------

def basic_diagnostics(df: pd.DataFrame) -> dict:
    """Return a global overview of the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "empty_columns": int((df.isna().all()).sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "text_columns": int(df.select_dtypes(include="object").shape[1]),
        "datetime_columns": int(df.select_dtypes(include="datetime").shape[1]),
    }


def quality_score(df: pd.DataFrame) -> int:
    """
    Return a data quality score from 0 to 100.
    Penalizes missing values, duplicates and empty columns.
    """
    total_cells = df.size if df.size > 0 else 1
    missing_ratio = df.isna().sum().sum() / total_cells
    duplicated_ratio = df.duplicated().sum() / max(len(df), 1)
    empty_columns_ratio = (df.isna().all()).sum() / max(df.shape[1], 1)

    score = 100
    score -= missing_ratio * 50
    score -= duplicated_ratio * 30
    score -= empty_columns_ratio * 20
    return max(0, int(round(score)))


# -----------------------------
# COLUMN-LEVEL DIAGNOSTICS
# -----------------------------

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per column with useful statistics:
    type, missing count, missing %, unique values, sample values.
    """
    rows = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append({
            "column": column,
            "type": str(series.dtype),
            "missing": missing,
            "missing_pct": round(missing / len(df) * 100, 2) if len(df) else 0,
            "unique_values": int(series.nunique(dropna=True)),
            "sample_values": series.dropna().unique()[:3].tolist(),
        })
    return pd.DataFrame(rows)


# -----------------------------
# MISSING VALUES
# -----------------------------

def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns that contain missing values, sorted by count."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "column": missing.index,
        "missing": missing.values,
        "missing_pct": (missing.values / len(df) * 100).round(2),
    })


# -----------------------------
# DUPLICATES
# -----------------------------

def duplicates_preview(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return a preview of duplicated rows."""
    return df[df.duplicated(keep=False)].head(limit)


# -----------------------------
# OUTLIERS
# -----------------------------

def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> dict:
    """
    Detect outliers in a numeric column using the IQR method.
    Returns statistics and a preview of the outlier rows.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"is_numeric": False}

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    return {
        "is_numeric": True,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(df) * 100, 2) if len(df) else 0,
        "preview": outliers.head(10),
    }


def outliers_overview(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Return an outlier summary for all numeric columns."""
    rows = []
    for column in df.select_dtypes(include="number").columns:
        result = detect_outliers_iqr(df, column, factor=factor)
        rows.append({
            "column": column,
            "outlier_count": result["outlier_count"],
            "outlier_pct": result["outlier_pct"],
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
        })
    return pd.DataFrame(rows)


# -----------------------------
# NUMERIC STATISTICS
# -----------------------------

def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    stats = numeric_df.describe().T
    stats["missing"] = numeric_df.isna().sum().values
    return stats.reset_index().rename(columns={"index": "column"})


# -----------------------------
# TEXT STATISTICS
# -----------------------------

def text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple statistics for text columns."""
    text_df = df.select_dtypes(include="object")
    rows = []
    for column in text_df.columns:
        series = text_df[column].dropna().astype(str)
        rows.append({
            "column": column,
            "unique_values": int(series.nunique()),
            "most_common": series.mode().iloc[0] if not series.mode().empty else None,
            "avg_length": round(series.str.len().mean(), 2) if not series.empty else 0,
            "min_length": int(series.str.len().min()) if not series.empty else 0,
            "max_length": int(series.str.len().max()) if not series.empty else 0,
        })
    return pd.DataFrame(rows)
