"""
cleaner.py
Cleaning functions for the Smart Data Cleaner app.

Rules:
- Each function does ONE thing.
- Each function receives a DataFrame and returns a new DataFrame.
- Functions never modify the original DataFrame in place.
"""

import pandas as pd
import numpy as np


# -----------------------------
# COLUMNS
# -----------------------------

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop the given columns from the DataFrame."""
    return df.drop(columns=columns, errors="ignore")


def rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns using a mapping like {'old_name': 'new_name'}."""
    return df.rename(columns=mapping)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase names, strip spaces, replace spaces with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def convert_column_type(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    """
    Convert a column to a given type.
    Supported types: 'string', 'integer', 'float', 'boolean', 'datetime'.
    """
    df = df.copy()
    if new_type == "string":
        df[column] = df[column].astype(str)
    elif new_type == "integer":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    elif new_type == "float":
        df[column] = pd.to_numeric(df[column], errors="coerce")
    elif new_type == "boolean":
        df[column] = df[column].astype(bool)
    elif new_type == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


# -----------------------------
# ROWS
# -----------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows."""
    return df.drop_duplicates()


def remove_rows_with_missing_values(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Remove rows containing NaN, optionally only in specific columns."""
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()


def filter_rows(df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
    """
    Keep rows where the condition is true.
    Supported operators: '==', '!=', '>', '<', '>=', '<=', 'contains'.
    """
    if operator == "==":
        return df[df[column] == value]
    if operator == "!=":
        return df[df[column] != value]
    if operator == ">":
        return df[df[column] > value]
    if operator == "<":
        return df[df[column] < value]
    if operator == ">=":
        return df[df[column] >= value]
    if operator == "<=":
        return df[df[column] <= value]
    if operator == "contains":
        return df[df[column].astype(str).str.contains(str(value), na=False)]
    return df


# -----------------------------
# MISSING VALUES
# -----------------------------

def fill_missing_values(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    custom_value=None,
) -> pd.DataFrame:
    """
    Fill missing values in a specific column.
    Supported strategies:
    - 'median'    (numeric only)
    - 'mean'      (numeric only)
    - 'mode'
    - 'zero'
    - 'unknown'   (text)
    - 'custom'    (uses custom_value)
    - 'null'      (leave as NaN, no change)
    """
    df = df.copy()

    if strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "mode":
        mode_value = df[column].mode(dropna=True)
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])
    elif strategy == "zero":
        df[column] = df[column].fillna(0)
    elif strategy == "unknown":
        df[column] = df[column].fillna("Unknown")
    elif strategy == "custom" and custom_value is not None:
        df[column] = df[column].fillna(custom_value)
    elif strategy == "null":
        pass  # no change

    return df


def fill_all_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill all missing values with default strategy:
    - numeric columns → median
    - text columns → 'Unknown'
    """
    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df = fill_missing_values(df, column, strategy="median")
        else:
            df = fill_missing_values(df, column, strategy="unknown")
    return df


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns where all values are missing."""
    return df.dropna(axis=1, how="all")


# -----------------------------
# TEXT CLEANING
# -----------------------------

def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing spaces from all string columns."""
    df = df.copy()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def standardize_text_case(df: pd.DataFrame, mode: str = "lower") -> pd.DataFrame:
    """
    Standardize text case for all string columns.
    Modes: 'lower', 'upper', 'title'.
    """
    df = df.copy()
    for column in df.select_dtypes(include="object").columns:
        if mode == "lower":
            df[column] = df[column].astype(str).str.lower()
        elif mode == "upper":
            df[column] = df[column].astype(str).str.upper()
        elif mode == "title":
            df[column] = df[column].astype(str).str.title()
    return df


def replace_values(df: pd.DataFrame, column: str, to_replace, new_value) -> pd.DataFrame:
    """Replace a specific value inside a column."""
    df = df.copy()
    df[column] = df[column].replace(to_replace, new_value)
    return df


# -----------------------------
# OUTLIERS
# -----------------------------

def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers in a numeric column using the IQR method.
    Rows with values outside [Q1 - factor*IQR, Q3 + factor*IQR] are removed.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


# -----------------------------
# DATES
# -----------------------------

def parse_dates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a column to datetime, invalid values become NaT."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def extract_date_parts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Extract year, month and day from a datetime column.
    Adds three new columns: <column>_year, <column>_month, <column>_day.
    """
    df = df.copy()
    dates = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_year"] = dates.dt.year
    df[f"{column}_month"] = dates.dt.month
    df[f"{column}_day"] = dates.dt.day
    return df


# -----------------------------
# GLOBAL CLEANING
# -----------------------------

def express_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a default 'express' cleaning:
    - Standardize column names
    - Trim whitespace
    - Remove duplicated rows
    - Remove empty columns
    - Fill missing values (median for numeric, 'Unknown' for text)
    """
    df = standardize_column_names(df)
    df = trim_whitespace(df)
    df = remove_duplicates(df)
    df = remove_empty_columns(df)
    df = fill_all_missing_values(df)
    return df
