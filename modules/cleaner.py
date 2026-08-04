import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows."""
    return df.drop_duplicates()


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns where all values are missing."""
    return df.dropna(axis=1, how="all")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, 'Unknown' for text."""
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna("Unknown")
    return df
