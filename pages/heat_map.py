# """
# pages/heat_map.py — Visualize all raster layers for Patna
# """

# import streamlit as st
# import numpy as np
# import rasterio
# import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd

# from config import (
#     LST_PATH, NDVI_PATH, LULC_PATH,
#     ALBEDO_PATH, NDWI_PATH, UHII_PATH,
#     LULC_LABELS
# )

# st.set_page_config(page_title="Heat Map — UrbanCool AI", page_icon="🗺️", layout="wide")

# st.title("🗺️ Patna Heat Maps")
# st.markdown("Spatial view of temperature and its physical drivers across the city")
# st.markdown("---")

# # ── LOAD RASTERS (cached) ─────────────────────
# @st.cache_data
# def load_raster(path):
#     with rasterio.open(path) as src:
#         data = src.read(1).astype(float)
#         if src.nodata is not None:
#             data[data == src.nodata] = np.nan
#     return data

# with st.spinner("Loading raster layers..."):
#     lst    = load_raster(LST_PATH)
#     ndvi   = load_raster(NDVI_PATH)
#     lulc   = load_raster(LULC_PATH)
#     albedo = load_raster(ALBEDO_PATH)
#     ndwi   = load_raster(NDWI_PATH)
#     uhii   = load_raster(UHII_PATH)

# # Downsampled copies — used ONLY for plotting, never for stats.
# # Keeps memory low when rendering multiple large images at once.
# DOWNSAMPLE = 4

# def downsample(arr, factor=DOWNSAMPLE):
#     return arr[::factor, ::factor]

# lst_ds    = downsample(lst)
# ndvi_ds   = downsample(ndvi)
# lulc_ds   = downsample(lulc)
# albedo_ds = downsample(albedo)
# ndwi_ds   = downsample(ndwi)
# uhii_ds   = downsample(uhii)

# # ── LAYER SELECTOR ────────────────────────────
# layer_choice = st.selectbox(
#     "Select layer to view large",
#     ["Land Surface Temperature (LST)", "NDVI (Vegetation)",
#      "LULC (Land Use)", "Albedo (Reflectivity)",
#      "NDWI (Water)", "UHII (Heat Island Intensity)"]
# )

# layer_map = {
#     "Land Surface Temperature (LST)": (lst_ds, lst, "RdYlGn_r", "°C"),
#     "NDVI (Vegetation)":              (ndvi_ds, ndvi, "RdYlGn", "NDVI"),
#     "LULC (Land Use)":                (lulc_ds, lulc, "tab10", "Class"),
#     "Albedo (Reflectivity)":          (albedo_ds, albedo, "gray", "Albedo"),
#     "NDWI (Water)":                   (ndwi_ds, ndwi, "RdBu", "NDWI"),
#     "UHII (Heat Island Intensity)":   (uhii_ds, uhii, "hot", "°C above rural"),
# }

# # plot_data = downsampled (fast to render), stats_data = full resolution (accurate)
# plot_data, stats_data, cmap, label = layer_map[layer_choice]

# col1, col2 = st.columns([3, 1])

# with col1:
#     fig, ax = plt.subplots(figsize=(9, 7))
#     im = ax.imshow(plot_data, cmap=cmap)
#     plt.colorbar(im, ax=ax, label=label)
#     ax.set_title(layer_choice)
#     ax.axis("off")
#     st.pyplot(fig)
#     plt.close(fig)

# with col2:
#     st.markdown("**Quick stats** (full resolution)")
#     st.metric("Mean", f"{np.nanmean(stats_data):.2f}")
#     st.metric("Max",  f"{np.nanmax(stats_data):.2f}")
#     st.metric("Min",  f"{np.nanmin(stats_data):.2f}")
#     st.metric("Std Dev", f"{np.nanstd(stats_data):.2f}")

# st.markdown("---")

# # ── ALL 6 LAYERS SIDE BY SIDE ─────────────────
# st.subheader("All Layers Overview")

# fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# layers_grid = [
#     (lst_ds, "RdYlGn_r", "LST (°C)"),
#     (ndvi_ds, "RdYlGn", "NDVI"),
#     (lulc_ds, "tab10", "LULC"),
#     (albedo_ds, "gray", "Albedo"),
#     (ndwi_ds, "RdBu", "NDWI"),
#     (uhii_ds, "hot", "UHII (°C)"),
# ]
# for ax, (arr, cm, title) in zip(axes.flat, layers_grid):
#     im = ax.imshow(arr, cmap=cm)
#     ax.set_title(title, fontsize=11)
#     ax.axis("off")
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# st.pyplot(fig)
# plt.close(fig)

# st.markdown("---")

# # ── LST DISTRIBUTION + HOTSPOT THRESHOLD ─────
# st.subheader("Temperature Distribution")

# valid_lst = lst[~np.isnan(lst)]
# hot_threshold  = np.percentile(valid_lst, 90)
# cool_threshold = np.percentile(valid_lst, 10)

# # Sample for the histogram — stats above use full data, this is just for the chart
# sample_size = min(200_000, len(valid_lst))
# rng = np.random.default_rng(42)
# lst_sample = rng.choice(valid_lst, size=sample_size, replace=False)

# fig = px.histogram(
#     pd.DataFrame({"LST": lst_sample}), x="LST", nbins=60,
#     color_discrete_sequence=["tomato"],
#     title="Land Surface Temperature Distribution — Patna"
# )
# fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red",
#               annotation_text=f"Hotspot > {hot_threshold:.1f}°C")
# fig.add_vline(x=cool_threshold, line_dash="dash", line_color="blue",
#               annotation_text=f"Cool < {cool_threshold:.1f}°C")
# fig.update_layout(xaxis_title="LST (°C)", yaxis_title="Pixel count")
# st.plotly_chart(fig, use_container_width=True)

# col1, col2, col3 = st.columns(3)
# col1.metric("🔴 Hotspot threshold", f"> {hot_threshold:.1f}°C")
# col2.metric("🔵 Cool zone threshold", f"< {cool_threshold:.1f}°C")
 # col3.metric("⚠️ Hotspot pixels", f"{int((valid_lst >= hot_threshold).sum()):,}")


# //////////////////////////////////////////////////////////////////////////////////

"""
pages/heat_map.py — Visualize all raster layers for Patna
(Memory-optimized: loads one raster at a time, float32, cached stats/figures)
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from config import (
    LST_PATH, NDVI_PATH, LULC_PATH,
    ALBEDO_PATH, NDWI_PATH, UHII_PATH,
    LULC_LABELS
)

st.set_page_config(page_title="Heat Map — UrbanCool AI", page_icon="🗺️", layout="wide")

st.title("🗺️ Patna Heat Maps")
st.markdown("Spatial view of temperature and its physical drivers across the city")
st.markdown("---")

DOWNSAMPLE = 4

# ── LOAD A SINGLE RASTER (cached, float32) ────
@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)  # float32 instead of float64: half the memory
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data


def downsample(arr, factor=DOWNSAMPLE):
    return arr[::factor, ::factor]


# ── LAYER SELECTOR ────────────────────────────
layer_paths = {
    "Land Surface Temperature (LST)": (LST_PATH, "RdYlGn_r", "°C"),
    "NDVI (Vegetation)":              (NDVI_PATH, "RdYlGn", "NDVI"),
    "LULC (Land Use)":                (LULC_PATH, "tab10", "Class"),
    "Albedo (Reflectivity)":          (ALBEDO_PATH, "gray", "Albedo"),
    "NDWI (Water)":                   (NDWI_PATH, "RdBu", "NDWI"),
    "UHII (Heat Island Intensity)":   (UHII_PATH, "hot", "°C above rural"),
}

layer_choice = st.selectbox("Select layer to view large", list(layer_paths.keys()))
path, cmap, label = layer_paths[layer_choice]

# Only ONE raster loaded fully at a time for this section
with st.spinner(f"Loading {layer_choice}..."):
    stats_data = load_raster(path)
plot_data = downsample(stats_data)

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(plot_data, cmap=cmap)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(layer_choice)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("**Quick stats** (full resolution)")
    st.metric("Mean", f"{np.nanmean(stats_data):.2f}")
    st.metric("Max",  f"{np.nanmax(stats_data):.2f}")
    st.metric("Min",  f"{np.nanmin(stats_data):.2f}")
    st.metric("Std Dev", f"{np.nanstd(stats_data):.2f}")

st.markdown("---")

# ── ALL 6 LAYERS SIDE BY SIDE ─────────────────
# Built ONCE and cached as a resource — doesn't depend on layer_choice,
# so it should never be rebuilt just because the dropdown changed.
# Loads/downsamples/discards one raster at a time instead of holding all 6.
st.subheader("All Layers Overview")

@st.cache_resource
def build_overview_figure():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    grid_specs = [
        (LST_PATH, "RdYlGn_r", "LST (°C)"),
        (NDVI_PATH, "RdYlGn", "NDVI"),
        (LULC_PATH, "tab10", "LULC"),
        (ALBEDO_PATH, "gray", "Albedo"),
        (NDWI_PATH, "RdBu", "NDWI"),
        (UHII_PATH, "hot", "UHII (°C)"),
    ]
    for ax, (p, cm, title) in zip(axes.flat, grid_specs):
        arr = downsample(load_raster(p))
        im = ax.imshow(arr, cmap=cm)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

overview_fig = build_overview_figure()
st.pyplot(overview_fig)
# Note: do NOT plt.close() a cache_resource figure — Streamlit needs to
# keep reusing the same cached object across reruns.

st.markdown("---")

# ── LST DISTRIBUTION + HOTSPOT THRESHOLD ─────
# Cached so percentile calc + 200k sample draw only happens once total,
# not every time the layer dropdown changes.
st.subheader("Temperature Distribution")

@st.cache_data
def get_lst_distribution_data():
    lst = load_raster(LST_PATH)
    valid_lst = lst[~np.isnan(lst)]
    hot_threshold = np.percentile(valid_lst, 90)
    cool_threshold = np.percentile(valid_lst, 10)
    sample_size = min(200_000, len(valid_lst))
    rng = np.random.default_rng(42)
    lst_sample = rng.choice(valid_lst, size=sample_size, replace=False)
    hotspot_count = int((valid_lst >= hot_threshold).sum())
    return hot_threshold, cool_threshold, lst_sample, hotspot_count

hot_threshold, cool_threshold, lst_sample, hotspot_count = get_lst_distribution_data()

fig = px.histogram(
    pd.DataFrame({"LST": lst_sample}), x="LST", nbins=60,
    color_discrete_sequence=["tomato"],
    title="Land Surface Temperature Distribution — Patna"
)
fig.add_vline(x=hot_threshold, line_dash="dash", line_color="red",
              annotation_text=f"Hotspot > {hot_threshold:.1f}°C")
fig.add_vline(x=cool_threshold, line_dash="dash", line_color="blue",
              annotation_text=f"Cool < {cool_threshold:.1f}°C")
fig.update_layout(xaxis_title="LST (°C)", yaxis_title="Pixel count")
st.plotly_chart(fig, width="stretch")  # use_container_width is deprecated

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Hotspot threshold", f"> {hot_threshold:.1f}°C")
col2.metric("🔵 Cool zone threshold", f"< {cool_threshold:.1f}°C")
col3.metric("⚠️ Hotspot pixels", f"{hotspot_count:,}")
