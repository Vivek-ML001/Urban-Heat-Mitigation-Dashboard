"""
app.py — UrbanCool AI home page
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import rasterio
import os
from config import LST_PATH, NDVI_PATH, UHII_PATH, DATA_DIR

st.set_page_config(
    page_title="UrbanCool AI",
    page_icon="🌡️",
    layout="wide"
)

# ── HEADER ────────────────────────────────────
st.title("🌡️ UrbanCool AI")
st.subheader("Urban Heat Mitigation Dashboard — Patna")

st.markdown("---")

# ── CHECK DATA AVAILABILITY ──────────────────
def file_exists(path):
    return os.path.exists(path)

files_status = {
    "LST":  LST_PATH,
    "NDVI": NDVI_PATH,
    "UHII": UHII_PATH,
}

missing = [name for name, path in files_status.items() if not file_exists(path)]

if missing:
    st.error(
        f"⚠️ Missing data files: {', '.join(missing)}\n\n"
        f"Expected folder: `{DATA_DIR}`\n\n"
        f"Copy your .tif files there, then refresh this page."
    )
    st.stop()

# ── QUICK STATS (loaded once, cached) ────────
@st.cache_data
def load_quick_stats():
    with rasterio.open(LST_PATH) as src:
        lst = src.read(1).astype(float)
        if src.nodata is not None:
            lst[lst == src.nodata] = np.nan

    with rasterio.open(NDVI_PATH) as src:
        ndvi = src.read(1).astype(float)
        if src.nodata is not None:
            ndvi[ndvi == src.nodata] = np.nan

    return {
        "mean_lst": np.nanmean(lst),
        "max_lst": np.nanmax(lst),
        "min_lst": np.nanmin(lst),
        "mean_ndvi": np.nanmean(ndvi),
        "valid_pixels": int(np.sum(~np.isnan(lst)))
    }

with st.spinner("Loading Patna heat data..."):
    stats = load_quick_stats()

# ── METRICS ROW ───────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Mean Temperature", f"{stats['mean_lst']:.1f}°C")
col2.metric("🔴 Max Temperature",  f"{stats['max_lst']:.1f}°C")
col3.metric("🔵 Min Temperature",  f"{stats['min_lst']:.1f}°C")
col4.metric("🌿 Mean NDVI",        f"{stats['mean_ndvi']:.3f}")

st.markdown("---")

# ── ABOUT SECTION ─────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### What this app does

    UrbanCool AI identifies urban heat stress hotspots in Patna using
    satellite data and machine learning, then simulates cooling
    interventions to recommend where the city should act first.

    **Use the pages in the sidebar to explore:**
    - 🗺️ **Heat Map** — Land surface temperature across Patna
    - 📊 **Driver Analysis** — What causes heat in each area
    - 🌿 **Scenario Simulator** — Test cooling interventions and see predicted °C reduction
    """)

with col2:
    st.info(
        f"**Data coverage**\n\n"
        f"Valid pixels analyzed: {stats['valid_pixels']:,}\n\n"
        f"Resolution: 30m × 30m\n\n"
        f"Season: Summer 2023"
    )

st.markdown("---")
st.caption("Built with Streamlit, XGBoost, and Google Earth Engine data | Landsat 8 · ESA WorldCover · ERA5")