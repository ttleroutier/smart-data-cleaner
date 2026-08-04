#Voici une version enrichie de `app.py`, en anglais, structurée par sections claires, avec le principe **"See before you act"** appliqué à chaque action.

#Ce fichier reste lisible car il **délègue toute la logique** à `diagnostics.py` et `cleaner.py`. Il ne contient que l’interface.

#```python
#"""
#app.py
#Smart Data Cleaner - main Streamlit interface.

#Structure:
#- Sidebar: upload + navigation
#- Main area: one section per cleaning action
#- Each section shows STATISTICS first, then applies changes
#"""

import streamlit as st
import pandas as pd

from modules.loader import load_file
from modules import diagnostics as diag
from modules import cleaner as clean


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(page_title="Smart Data Cleaner", layout="wide")
st.title("🧼 Smart Data Cleaner")
st.caption("Clean your data step by step. See statistics before you apply any change.")


# -----------------------------
# SESSION STATE
# -----------------------------

if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# SIDEBAR: FILE UPLOAD
# -----------------------------

st.sidebar.header("1. Upload your file")
uploaded_file = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and st.session_state.df is None:
    df = load_file(uploaded_file)
    st.session_state.df = df.copy()
    st.session_state.original_df = df.copy()

if st.session_state.df is None:
    st.info("Upload a file in the sidebar to start.")
    st.stop()


df = st.session_state.df


# -----------------------------
# SIDEBAR: NAVIGATION
# -----------------------------

st.sidebar.header("2. Choose an action")
action = st.sidebar.radio(
    "Cleaning steps",
    [
        "Overview",
        "Columns",
        "Duplicates",
        "Missing values",
        "Outliers",
        "Text cleaning",
        "Dates",
        "Express cleaning",
        "History",
        "Compare & Download",
    ],
)



# -----------------------------
# HELPERS
# -----------------------------

from datetime import datetime


def apply_change(new_df: pd.DataFrame, message: str):
    """Save the modified DataFrame, keep history, and show a message."""
    # Save previous state in history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "description": message,
        "snapshot": st.session_state.df.copy(),
    })

    st.session_state.df = new_df
    st.success(message)
    st.rerun()

# -----------------------------
# OVERVIEW
# -----------------------------

if action == "Overview":
    st.subheader("Dataset overview")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Quality score", f"{diag.quality_score(df)} / 100")
        st.json(diag.basic_diagnostics(df))
    with col2:
        st.write("Preview:")
        st.dataframe(df.head(10))

    st.subheader("Column summary")
    st.dataframe(diag.column_summary(df))


# -----------------------------
# COLUMNS
# -----------------------------

elif action == "Columns":
    st.subheader("Manage columns")

    st.write("Current columns:")
    st.dataframe(diag.column_summary(df))

    st.markdown("### Drop columns")
    cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
    if cols_to_drop and st.button("Apply: drop columns"):
        apply_change(clean.drop_columns(df, cols_to_drop),
                     f"Dropped {len(cols_to_drop)} column(s).")

    st.markdown("### Standardize column names")
    st.caption("Lowercase, no spaces, underscores instead of dashes.")
    if st.button("Apply: standardize names"):
        apply_change(clean.standardize_column_names(df),
                     "Column names standardized.")

    st.markdown("### Convert a column type")
    col = st.selectbox("Column", df.columns.tolist(), key="convert_col")
    new_type = st.selectbox(
        "New type",
        ["string", "integer", "float", "boolean", "datetime"],
    )
    st.caption(f"Current type: {df[col].dtype}")
    if st.button("Apply: convert type"):
        apply_change(clean.convert_column_type(df, col, new_type),
                     f"Converted '{col}' to {new_type}.")


# -----------------------------
# DUPLICATES
# -----------------------------

elif action == "Duplicates":
    st.subheader("Duplicated rows")

    duplicated_count = int(df.duplicated().sum())
    st.metric("Duplicated rows detected", duplicated_count)

    if duplicated_count > 0:
        st.write("Preview of duplicated rows:")
        st.dataframe(diag.duplicates_preview(df))

        if st.button("Apply: remove duplicates"):
            apply_change(clean.remove_duplicates(df),
                         f"Removed {duplicated_count} duplicated row(s).")
    else:
        st.success("No duplicated rows found.")


# -----------------------------
# MISSING VALUES
# -----------------------------

elif action == "Missing values":
    st.subheader("Missing values")

    report = diag.missing_values_report(df)
    if report.empty:
        st.success("No missing values in the dataset.")
    else:
        st.write("Columns with missing values:")
        st.dataframe(report)

        st.markdown("### Fill missing values in a column")
        col = st.selectbox("Column", report["column"].tolist())
        strategy = st.selectbox(
            "Strategy",
            ["median", "mean", "mode", "zero", "unknown", "custom", "null"],
        )
        custom_value = None
        if strategy == "custom":
            custom_value = st.text_input("Custom value")

        # Show what will happen
        missing_count = int(df[col].isna().sum())
        st.caption(f"Missing values in '{col}': {missing_count}")
        st.write("Preview of rows with missing values:")
        st.dataframe(df[df[col].isna()].head(5))

        if st.button("Apply: fill missing values"):
            apply_change(
                clean.fill_missing_values(df, col, strategy, custom_value),
                f"Filled missing values in '{col}' using '{strategy}'.",
            )

        st.markdown("### Remove rows with missing values")
        cols_for_dropna = st.multiselect(
            "Only consider these columns (optional)",
            df.columns.tolist(),
        )
        rows_to_remove = df[df[cols_for_dropna].isna().any(axis=1)] if cols_for_dropna else df[df.isna().any(axis=1)]
        st.caption(f"Rows that will be removed: {len(rows_to_remove)}")
        st.dataframe(rows_to_remove.head(5))
        if st.button("Apply: remove rows with missing values"):
            apply_change(
                clean.remove_rows_with_missing_values(df, cols_for_dropna or None),
                "Removed rows with missing values.",
            )

        st.markdown("### Remove fully empty columns")
        empty_cols = df.columns[df.isna().all()].tolist()
        st.caption(f"Empty columns detected: {empty_cols or 'None'}")
        if empty_cols and st.button("Apply: remove empty columns"):
            apply_change(clean.remove_empty_columns(df),
                         f"Removed {len(empty_cols)} empty column(s).")


# -----------------------------
# OUTLIERS
# -----------------------------

elif action == "Outliers":
    st.subheader("Outlier detection")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        st.write("Overview of outliers per column:")
        st.dataframe(diag.outliers_overview(df))

        st.markdown("### Detect outliers in a specific column")
        col = st.selectbox("Column", numeric_cols)
        factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)

        result = diag.detect_outliers_iqr(df, col, factor)
        st.write({
            "Q1": result["q1"],
            "Q3": result["q3"],
            "Lower bound": result["lower_bound"],
            "Upper bound": result["upper_bound"],
            "Outlier count": result["outlier_count"],
            "Outlier %": result["outlier_pct"],
        })

        st.write("Preview of outlier rows:")
        st.dataframe(result["preview"])

        st.bar_chart(df[col])

        if result["outlier_count"] > 0 and st.button("Apply: remove outliers"):
            apply_change(
                clean.remove_outliers_iqr(df, col, factor),
                f"Removed {result['outlier_count']} outlier row(s) from '{col}'.",
            )


# -----------------------------
# TEXT CLEANING
# -----------------------------

elif action == "Text cleaning":
    st.subheader("Text cleaning")

    st.dataframe(diag.text_statistics(df))

    st.markdown("### Trim whitespace")
    if st.button("Apply: trim whitespace"):
        apply_change(clean.trim_whitespace(df), "Whitespace trimmed.")

    st.markdown("### Standardize text case")
    mode = st.selectbox("Case", ["lower", "upper", "title"])
    if st.button("Apply: standardize case"):
        apply_change(clean.standardize_text_case(df, mode),
                     f"Text case standardized to {mode}.")

    st.markdown("### Replace a value in a column")
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        col = st.selectbox("Column", text_cols, key="replace_col")
        to_replace = st.text_input("Value to replace")
        new_value = st.text_input("New value")
        occurrences = int((df[col].astype(str) == to_replace).sum()) if to_replace else 0
        st.caption(f"Occurrences found: {occurrences}")
        if to_replace and st.button("Apply: replace value"):
            apply_change(clean.replace_values(df, col, to_replace, new_value),
                         f"Replaced '{to_replace}' with '{new_value}' in '{col}'.")


# -----------------------------
# DATES
# -----------------------------

elif action == "Dates":
    st.subheader("Date handling")

    col = st.selectbox("Column to parse as date", df.columns.tolist())
    preview = pd.to_datetime(df[col], errors="coerce")
    invalid = int(preview.isna().sum())
    st.caption(f"Values that cannot be parsed as dates: {invalid}")
    st.dataframe(preview.head(5))

    if st.button("Apply: parse as datetime"):
        apply_change(clean.parse_dates(df, col), f"Converted '{col}' to datetime.")

    if st.button("Apply: extract year / month / day"):
        apply_change(clean.extract_date_parts(df, col),
                     f"Extracted date parts from '{col}'.")


# -----------------------------
# EXPRESS CLEANING
# -----------------------------

elif action == "Express cleaning":
    st.subheader("Express cleaning (1 click)")
    st.write("This will apply the following actions:")
    st.markdown("""
    - Standardize column names
    - Trim whitespace
    - Remove duplicated rows
    - Remove empty columns
    - Fill missing values (median for numeric, 'Unknown' for text)
    """)

    if st.button("Apply express cleaning"):
        apply_change(clean.express_clean(df), "Express cleaning applied.")


# -----------------------------
# HISTORY
# -----------------------------

elif action == "History":
    st.subheader("Action history")

    if not st.session_state.history:
        st.info("No actions have been applied yet.")
    else:
        st.write(f"Total actions applied: {len(st.session_state.history)}")

        # Show history table
        history_df = pd.DataFrame([
            {"step": i + 1, "time": h["timestamp"], "action": h["description"]}
            for i, h in enumerate(st.session_state.history)
        ])
        st.dataframe(history_df, use_container_width=True)

        st.markdown("### Undo the last action")
        if st.button("↩ Undo last action"):
            last = st.session_state.history.pop()
            st.session_state.df = last["snapshot"]
            st.success(f"Reverted: {last['description']}")
            st.rerun()

        st.markdown("### Revert to a specific step")
        step = st.number_input(
            "Go back to the dataset BEFORE this step number",
            min_value=1,
            max_value=len(st.session_state.history),
            step=1,
        )
        if st.button("↩ Revert to this step"):
            target = st.session_state.history[step - 1]
            # Restore the snapshot taken before that step
            st.session_state.df = target["snapshot"]
            # Truncate history so future = clean
            st.session_state.history = st.session_state.history[: step - 1]
            st.success(f"Reverted to state before step {step}.")
            st.rerun()

        st.markdown("### Reset everything")
        if st.button("🔄 Reset to original dataset"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.history = []
            st.success("All actions cleared. Dataset reset to original.")
            st.rerun()

# -----------------------------
# COMPARE & DOWNLOAD
# -----------------------------

elif action == "Compare & Download":
    st.subheader("Before / After comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original dataset**")
        st.json(diag.basic_diagnostics(st.session_state.original_df))
        st.dataframe(st.session_state.original_df.head(5))
    with col2:
        st.markdown("**Cleaned dataset**")
        st.json(diag.basic_diagnostics(df))
        st.dataframe(df.head(5))

    st.subheader("Download cleaned dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    if st.button("🔄 Reset to original dataset"):
        st.session_state.df = st.session_state.original_df.copy()
        st.success("Dataset reset to original.")
        st.rerun()
```

