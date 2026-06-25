"""
pages/driver_analysis.py — What drives heat in Patna?
Correlation analysis, LST by land cover, and model feature importance.
"""

import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import joblib
import plotly.express as px
# import seaborn as sns
import matplotlib.pyplot as plt

from config import (
    LST_PATH, NDVI_PATH, LULC_PATH, ALBEDO_PATH, NDWI_PATH,
    MODEL_PATH, LULC_LABELS
)

st.set_page_config(page_title="Driver Analysis — UrbanCool AI", page_icon="📊", layout="wide")

st.title("📊 What Drives Heat in Patna?")
st.markdown("Correlation analysis and model-based feature importance")
st.markdown("---")

# ── LOAD RASTERS ────────────────────────────────
@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data

@st.cache_data
def load_raster_resampled(path, target_shape, _target_transform, _target_crs):
    """Resample a raster (e.g. LULC at 10m) onto the grid of the other
    30m layers, using nearest-neighbor since LULC values are categorical."""
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as src:
        dest = np.full(target_shape, np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=_target_transform,
            dst_crs=_target_crs,
            resampling=Resampling.nearest,
        )
    return dest

@st.cache_data
def build_feature_table():
    with rasterio.open(LST_PATH) as ref_src:
        target_shape = (ref_src.height, ref_src.width)
        target_transform = ref_src.transform
        target_crs = ref_src.crs

    lst    = load_raster(LST_PATH)
    ndvi   = load_raster(NDVI_PATH)
    albedo = load_raster(ALBEDO_PATH)
    ndwi   = load_raster(NDWI_PATH)
    lulc   = load_raster_resampled(LULC_PATH, target_shape, target_transform, target_crs)

    df = pd.DataFrame({
        "LST": lst.flatten(),
        "NDVI": ndvi.flatten(),
        "LULC": lulc.flatten(),
        "Albedo": albedo.flatten(),
        "NDWI": ndwi.flatten(),
    })

    df = df.dropna()
    df = df[df["LST"].between(20, 60)]
    df = df[df["NDVI"].between(-1, 1)]
    df = df[df["Albedo"].between(0, 1)]
    df["LULC_Name"] = df["LULC"].map(LULC_LABELS).fillna("Other")
    return df

with st.spinner("Loading Patna feature data..."):
    df = build_feature_table()

st.success(f"✅ Loaded {len(df):,} valid pixels for analysis")

# ── CORRELATION METRICS ─────────────────────────
st.subheader("Correlation with Land Surface Temperature")

corr_ndvi   = df["LST"].corr(df["NDVI"])
corr_albedo = df["LST"].corr(df["Albedo"])
corr_ndwi   = df["LST"].corr(df["NDWI"])

col1, col2, col3 = st.columns(3)
col1.metric("NDVI vs LST",   f"{corr_ndvi:.3f}",   "More trees → cooler 🌿" if corr_ndvi < 0 else "More trees → hotter")
col2.metric("Albedo vs LST", f"{corr_albedo:.3f}", "Brighter → cooler ☀️" if corr_albedo < 0 else "Brighter → hotter")
col3.metric("NDWI vs LST",   f"{corr_ndwi:.3f}",   "More water → cooler 💧" if corr_ndwi < 0 else "More water → hotter")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    corr_df = pd.DataFrame({
        "Feature": ["NDVI", "Albedo", "NDWI"],
        "Correlation": [corr_ndvi, corr_albedo, corr_ndwi]
    })
    fig = px.bar(
        corr_df, x="Feature", y="Correlation",
        color="Feature",
        color_discrete_map={"NDVI": "green", "Albedo": "orange", "NDWI": "blue"},
        title="Feature Correlation with LST"
    )
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    lulc_lst = df.groupby("LULC_Name")["LST"].mean().reset_index().sort_values("LST", ascending=False)
    fig = px.bar(
        lulc_lst, x="LULC_Name", y="LST",
        color="LST", color_continuous_scale="RdYlGn_r",
        title="Average LST by Land Cover Type"
    )
    fig.update_layout(xaxis_title="Land Cover", yaxis_title="Mean LST (°C)", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── FULL CORRELATION HEATMAP ─────────────────────
st.markdown("---")
st.subheader("Correlation Matrix")

corr_matrix = df[["LST", "NDVI", "Albedo", "NDWI"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True, ax=ax)
ax.set_title("Feature Correlation Matrix — Patna")
st.pyplot(fig)
plt.close(fig)

# ── SCATTER: NDVI vs LST ─────────────────────────
st.markdown("---")
st.subheader("🌿 NDVI vs LST Relationship")

sample = df.sample(min(3000, len(df)), random_state=42)
fig = px.scatter(
    sample, x="NDVI", y="LST", color="LULC_Name",
    opacity=0.5, title="NDVI vs Land Surface Temperature", trendline="ols"
)
fig.update_layout(xaxis_title="NDVI (Vegetation Index)", yaxis_title="LST (°C)")
st.plotly_chart(fig, use_container_width=True)

# ── MODEL FEATURE IMPORTANCE ─────────────────────
st.markdown("---")
st.subheader("🤖 Model Feature Importance")

try:
    model = joblib.load(MODEL_PATH)
    model_features = ["NDVI", "NDWI", "Albedo", "LULC"]  # must match training order

    imp_df = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Reds",
        title="What Drives LST Predictions? (XGBoost feature importance)"
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance reflects how often and how impactfully each variable "
        "was used to split decision trees inside the model — not the same as "
        "correlation, but usually points in a similar direction."
    )
except FileNotFoundError:
    st.warning(f"Model file not found at `{MODEL_PATH}` — feature importance unavailable.")
except Exception as e:
    st.warning(f"Could not load feature importance: {e}")
