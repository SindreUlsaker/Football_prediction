# File: src/ui_components/widgets.py
import streamlit as st
import pandas as pd

def round_input(min_value: int, max_value: int, value: int) -> int:
    return st.number_input(
        "Velg rundenummer",
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=1,
    )


def run_button(label: str) -> bool:
    return st.button(label)
