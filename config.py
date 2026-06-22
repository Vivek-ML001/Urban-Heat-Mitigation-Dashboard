"""
config.py — shared settings for the whole app
Every page imports from here so paths never break
"""

import os

# Base folder = wherever this project lives, on ANY computer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = os.path.join(BASE_DIR, "data", "Patna_Heat_Project")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data file paths
LST_PATH    = os.path.join(DATA_DIR, "Patna_LST_Summer2023.tif")
NDVI_PATH   = os.path.join(DATA_DIR, "Patna_NDVI_Summer2023.tif")
LULC_PATH   = os.path.join(DATA_DIR, "Patna_LULC_2021.tif")
ALBEDO_PATH = os.path.join(DATA_DIR, "Patna_Albedo_Summer2023.tif")
NDWI_PATH   = os.path.join(DATA_DIR, "Patna_NDWI_Summer2023.tif")
UHII_PATH   = os.path.join(DATA_DIR, "Patna_UHII_Summer2023.tif")
WEATHER_CSV = os.path.join(DATA_DIR, "Patna_ERA5_Weather_2023.csv")

MODEL_PATH  = os.path.join(MODEL_DIR, "Patna_Heat_Model.pkl")

LULC_LABELS = {
    10: "Forest", 20: "Shrubland", 30: "Grassland",
    40: "Cropland", 50: "Built-up", 60: "Bare soil",
    80: "Water", 95: "Wetland"
}