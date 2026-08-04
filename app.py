import streamlit as st
from modules.loader import load_file
from modules.diagnostics import basic_diagnostics

st.set_page_config(page_title="Smart Data Cleaner", layout="wide")

st.title("🧼 Smart Data Cleaner")
st.caption("Clean your data in 3 clicks, no code required.")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_file(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Diagnostics")
    diagnostics = basic_diagnostics(df)
    st.json(diagnostics)

