# File: app.py

import streamlit as st
import os
import sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

# ─── Må være det første Streamlit-kallet i hele appen ─────────────────────
st.set_page_config(
    page_title="Football Prediction App",
    layout="wide",
)
# ──────────────────────────────────────────────────────────────────────────

from src.ui_pages.main import main

if __name__ == "__main__":
    main()
