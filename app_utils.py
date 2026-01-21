import os
import pickle
import pandas as pd
import streamlit as st
import numpy as np

# NBER Recession Dates (approximate for FRED-MD plotting)
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

TRANSFORMATION_LABELS = {
    1: "Level",
    2: "Δ",
    3: "Δ²",
    4: "log",
    5: "Δlog",
    6: "Δ²log",
    7: "Δpct"
}

def save_engine_state(results, filename='engine_state.pkl'):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Failed to save engine state: {e}")

def load_engine_state(filename='engine_state.pkl'):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load engine state: {e}")
    return None

@st.cache_data(ttl=3600)
def get_series_descriptions(file_path: str = 'FRED-MD_updated_appendix.csv') -> dict:
    """Load series descriptions from appendix."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        # Create mapping from fred ID to description
        mapping = dict(zip(df['fred'], df['description']))
        # Add some manual mappings for derived variables if any
        mapping['SPREAD'] = '10Y Treasury - Fed Funds Spread'
        mapping['BAA_AAA'] = 'Baa - Aaa Corporate Bond Spread'
        mapping['CAPACITY'] = 'Capacity Utilization: Manufacturing'
        return mapping
    except Exception as e:
        st.warning(f"Could not load series descriptions: {e}")
        return {}

def get_transformation_label(tcode: int) -> str:
    """Get human-readable label for transformation code."""
    labels = {
        1: "Level (no change)",
        2: "Change: x(t) - x(t-1)",
        3: "Double Change",
        4: "Log: log(x(t))",
        5: "Log Change",
        6: "Double Log Change",
        7: "Change in % Change"
    }
    return labels.get(int(tcode), "Unknown")

def create_theme():
    theme_name = st.session_state.get('theme', 'dark')
    if theme_name == 'light':
        return {
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#ffffff',
            'gridcolor': '#f0f0f0',
            'linecolor': '#e0e0e0',
            'label_color': '#444444',
            'font': {'family': 'IBM Plex Mono', 'color': '#1a1a1a', 'size': 11},
            'xaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#e0e0e0', 'tickcolor': '#555', 'tickfont': {'color': '#666'}},
            'yaxis': {'gridcolor': '#f0f0f0', 'linecolor': '#e0e0e0', 'tickcolor': '#555', 'tickfont': {'color': '#666'}},
            'recession_color': '#cccccc',
            'border_color': '#dee2e6',
            'text_secondary': '#444444',
            'text_muted': '#666666'
        }
    else:
        return {
            'paper_bgcolor': '#0a0a0a',
            'plot_bgcolor': '#111111',
            'gridcolor': '#1a1a1a',
            'linecolor': '#2a2a2a',
            'label_color': '#888888',
            'font': {'family': 'IBM Plex Mono', 'color': '#e8e8e8', 'size': 11},
            'xaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a', 'tickcolor': '#888', 'tickfont': {'color': '#888'}},
            'yaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a', 'tickcolor': '#888', 'tickfont': {'color': '#888'}},
            'recession_color': '#ffffff',
            'border_color': '#2a2a2a',
            'text_secondary': '#888888',
            'text_muted': '#555555'
        }

@st.cache_data(ttl=3600)
def load_full_fred_md_raw(file_path: str = '2025-11-MD.csv') -> tuple:
    """Load the complete raw FRED-MD dataset and its transformation codes."""
    try:
        df_raw = pd.read_csv(file_path)
        transform_codes = df_raw.iloc[0]
        df = df_raw.iloc[1:].copy()
        df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True).dt.tz_localize(None)
        df = df.set_index('sasdate')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, transform_codes
    except Exception as e:
        st.error(f"Error loading full FRED-MD raw data: {e}")
        return pd.DataFrame(), pd.Series()

@st.cache_data(ttl=3600)
def load_fred_appendix(file_path: str = 'FRED-MD_updated_appendix.csv') -> pd.DataFrame:
    """Load FRED-MD appendix for series names and groupings."""
    try:
        # Try different encodings for robustness
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if 'fred' in df.columns:
                    # Normalize index to uppercase for robust matching
                    df['fred'] = df['fred'].str.upper()
                    df = df.set_index('fred')
                return df
            except UnicodeDecodeError:
                continue
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading FRED appendix: {e}")
        return pd.DataFrame()
