import streamlit as st


def section_title(title: str, subtitle: str = ""):
    """Render a section title with an optional subtitle."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

