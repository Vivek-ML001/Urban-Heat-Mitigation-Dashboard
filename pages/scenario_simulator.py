"""
pages/scenario_simulator.py — Cooling intervention "what-if" simulator
Uses the trained XGBoost model to predict LST under modified
NDVI / Albedo / NDWI values, simulating urban greening, cool roofs,
and water body interventions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

from config import (
    LST_PATH, NDVI_PATH, LULC_PATH, ALBEDO_PATH, NDWI_PATH,
    MODEL_PATH, LULC_LABELS
)

st.set_page_config(page_title="Scenario Simulator — UrbanCool AI", page_icon="🌿", layout="wide")

st.title(" Cooling Scenario Simulator")
st.markdown("Simulate urban greening, cool roofs, and water bodies — see predicted temperature reduction")
st.markdown("---")

# ── LOAD MODEL ─────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"⚠️ Model file not found at:\n\n`{MODEL_PATH}`\n\n"
        f"Copy your trained `Patna_Heat_Model.pkl` into the `models/` folder."
    )
    st.stop()

# ── LOAD RASTERS, BUILD FEATURE TABLE (cached) ─
@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data

@st.cache_data
def load_raster_resampled(path, target_shape, _target_transform, _target_crs):
    """For categorical rasters (like LULC) at a different resolution —
    resample onto the same grid as the other layers using nearest-neighbor
    (preserves class values, never invents fractional classes)."""
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

    # LULC is natively 10m (3x finer grid) — resample to match the
    # 30m Landsat grid used by everything else, nearest-neighbor only.
    lulc = load_raster_resampled(LULC_PATH, target_shape, target_transform, target_crs)

    shape = lst.shape

    df = pd.DataFrame({
        "LST": lst.flatten(),
        "NDVI": ndvi.flatten(),
        "LULC": lulc.flatten(),
        "Albedo": albedo.flatten(),
        "NDWI": ndwi.flatten(),
    })

    # Keep track of which flattened positions survive filtering,
    # so we can place results back into a 2D grid later.
    df["_pixel_id"] = np.arange(len(df))

    df = df.dropna()
    df = df[df["LST"].between(20, 60)]
    df = df[df["NDVI"].between(-1, 1)]
    df = df[df["Albedo"].between(0, 1)]

    df["LULC_Name"] = df["LULC"].map(LULC_LABELS).fillna("Other")
    return df, shape

with st.spinner("Loading Patna feature data..."):
    df, raster_shape = build_feature_table()

st.success(f"✅ Loaded {len(df):,} valid pixels for simulation")

FEATURES = ["NDVI", "NDWI", "Albedo", "LULC"]

expected_n = getattr(model, "n_features_in_", len(FEATURES))
if expected_n != len(FEATURES):
    st.warning(
        f"⚠️ Model was trained with {expected_n} features, but this page uses "
        f"{len(FEATURES)} ({', '.join(FEATURES)}). Predictions may be inaccurate "
        f"if the trained model expects a different feature set."
    )

# ── SIDEBAR CONTROLS ───────────────────────────
st.sidebar.header("🎛️ Intervention Intensity")
ndvi_increase   = st.sidebar.slider("🌿 NDVI increase (greening)", 0.0, 0.5, 0.25, 0.05)
albedo_increase = st.sidebar.slider("☀️ Albedo increase (cool roofs)", 0.0, 0.5, 0.35, 0.05)
ndwi_increase   = st.sidebar.slider("💧 NDWI increase (water bodies)", 0.0, 0.4, 0.20, 0.05)

available_classes = sorted(df["LULC"].dropna().unique())
target_class = st.sidebar.selectbox(
    "Apply interventions to which land cover?",
    options=available_classes,
    format_func=lambda x: f"{int(x)} — {LULC_LABELS.get(int(x), 'Other')}",
    index=0
)

# ── RUN SIMULATION ─────────────────────────────
target_mask = df["LULC"] == target_class

with st.spinner("Running baseline prediction..."):
    lst_baseline = model.predict(df[FEATURES])

def simulate(ndvi_add=0.0, albedo_add=0.0, ndwi_add=0.0):
    df_sim = df.copy()
    df_sim.loc[target_mask, "NDVI"]   = (df_sim.loc[target_mask, "NDVI"] + ndvi_add).clip(-1, 1)
    df_sim.loc[target_mask, "Albedo"] = (df_sim.loc[target_mask, "Albedo"] + albedo_add).clip(0, 1)
    df_sim.loc[target_mask, "NDWI"]   = (df_sim.loc[target_mask, "NDWI"] + ndwi_add).clip(-1, 1)
    lst_pred = model.predict(df_sim[FEATURES])
    return lst_pred - lst_baseline

with st.spinner("Simulating scenarios..."):
    delta_greening = simulate(ndvi_add=ndvi_increase)
    delta_roofs    = simulate(albedo_add=albedo_increase)
    delta_water    = simulate(ndwi_add=ndwi_increase)
    delta_combined = simulate(ndvi_increase, albedo_increase, ndwi_increase)

target_count = int(target_mask.sum())
st.info(f"📍 Interventions applied to **{target_count:,} pixels** classified as "
        f"**{LULC_LABELS.get(int(target_class), 'Other')}**")

if target_count == 0:
    st.warning("No pixels match this land cover class. Try a different selection.")
    st.stop()

# ── RESULTS TABLE ───────────────────────────────
st.markdown("---")
st.subheader("📊 Scenario Comparison")

results = pd.DataFrame({
    "Scenario": ["🌿 Urban Greening", "🏠 Cool Roofs", "💧 Water Bodies", "⭐ All Combined"],
    "Avg cooling (°C)": [
        round(abs(delta_greening[target_mask].mean()), 2),
        round(abs(delta_roofs[target_mask].mean()), 2),
        round(abs(delta_water[target_mask].mean()), 2),
        round(abs(delta_combined[target_mask].mean()), 2),
    ],
    "Max cooling (°C)": [
        round(abs(delta_greening[target_mask].min()), 2),
        round(abs(delta_roofs[target_mask].min()), 2),
        round(abs(delta_water[target_mask].min()), 2),
        round(abs(delta_combined[target_mask].min()), 2),
    ],
})

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌿 Greening",   f"−{results.iloc[0]['Avg cooling (°C)']}°C")
col2.metric("🏠 Cool Roofs", f"−{results.iloc[1]['Avg cooling (°C)']}°C")
col3.metric("💧 Water",      f"−{results.iloc[2]['Avg cooling (°C)']}°C")
col4.metric("⭐ Combined",   f"−{results.iloc[3]['Avg cooling (°C)']}°C")

fig = px.bar(
    results, x="Scenario", y="Avg cooling (°C)",
    color="Avg cooling (°C)", color_continuous_scale="Greens",
    text="Avg cooling (°C)", title="Average Cooling Effect per Scenario"
)
fig.update_traces(texttemplate="−%{text}°C", textposition="outside")
fig.update_layout(coloraxis_showscale=False, yaxis_title="Temperature reduction (°C)")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(results, use_container_width=True)

csv = results.to_csv(index=False)
st.download_button("⬇️ Download results CSV", csv, "Patna_Cooling_Results.csv", "text/csv")

# ── BEFORE / AFTER MAP (combined scenario) ─────
st.markdown("---")
st.subheader("🗺️ Before vs After — Combined Scenario")

DOWNSAMPLE = 4

def reshape_to_grid(values, shape, pixel_ids):
    """Place `values` back into a flat array of size shape[0]*shape[1]
    at positions given by `pixel_ids`, then reshape to 2D."""
    full = np.full(shape[0] * shape[1], np.nan)
    full[pixel_ids] = values
    return full.reshape(shape)

pixel_ids = df["_pixel_id"].values

with st.spinner("Building before/after maps..."):
    lst_before_2d = reshape_to_grid(lst_baseline, raster_shape, pixel_ids)
    delta_2d      = reshape_to_grid(delta_combined, raster_shape, pixel_ids)
    lst_after_2d  = lst_before_2d + delta_2d

def ds(arr, factor=DOWNSAMPLE):
    return arr[::factor, ::factor]

vmin = np.nanpercentile(lst_before_2d, 2)
vmax = np.nanpercentile(lst_before_2d, 98)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

im1 = axes[0].imshow(ds(lst_before_2d), cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
axes[0].set_title("BEFORE — Current LST (°C)")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0], fraction=0.046)

im2 = axes[1].imshow(ds(lst_after_2d), cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
axes[1].set_title("AFTER — Combined Scenario (°C)")
axes[1].axis("off")
plt.colorbar(im2, ax=axes[1], fraction=0.046)

im3 = axes[2].imshow(ds(delta_2d), cmap="RdBu_r", vmin=-5, vmax=5)
axes[2].set_title("CHANGE (°C, blue = cooler)")
axes[2].axis("off")
plt.colorbar(im3, ax=axes[2], fraction=0.046)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)