# SENTINEL 2-DATA-GROWTH-MONITORING USING VEGETATION INDICES AND BANDS
# Satellite-Based Crop Prediction in Haryana

A machine learning project to predict crop vigor from early-season Sentinel-2 satellite data, revealing the fundamental limits of satellite-only forecasting.

## Overview

This project investigates whether early-season satellite data can predict end-of-season crop vigor in Haryana's dual-cropping system. Using 4 seasons of Sentinel-2 imagery and 5 ML algorithms, we quantify predictability across Kharif (monsoon) and Rabi (winter) crops.

## Study Area

- **Location:** Haryana, India (Tile T43RFN)
- **Area:** ~12,000 km² (10980 × 10980 pixels at 10m)
- **Crops:** Wheat (Rabi), Rice/Cotton (Kharif)

## Dataset

### Satellite Data
| Source | Sentinel-2 (ESA Copernicus) |
|--------|----------------------------|
| Bands | B02 (blue), B03 (green), B04 (red), B08 (NIR) |
| Dates | Nov 2025, Jan 2026, Jun 2025, Sept 2025 |

### Features (16 per season)
- **Raw bands:** B02, B03, B04, B08
- **Indices:** NDVI, EVI, SAVI, GNDVI, VARI, NDWI, CI, NDVIre, MSAVI, ratio84, brightness, diff84

### Training Data
- 100,000 random samples from 120M total pixels
- 32 total features (16 × 2 seasons)

## Models Tested

| Algorithm | Library | Purpose |
|-----------|---------|---------|
| XGBoost | xgboost | Gradient boosting |
| Random Forest | scikit-learn | Ensemble of trees |
| Gradient Boosting | scikit-learn | Sequential boosting |
| Decision Tree | scikit-learn | Single tree baseline |
| Linear Regression | scikit-learn | Linear baseline |

## 📈 Results
### Spatial Pattern correlation:- 0.65
### Kharif (June → September)
| Model | R² Score |
|-------|----------|
| XGBoost | 0.178 |
| Gradient Boosting | 0.160 |
| Random Forest | 0.137 |
| Decision Tree | 0.115 |
| Linear Regression | 0.050 |

### Rabi (November → January)
| Model | R² Score |
|-------|----------|
| Random Forest | 0.34 |

## Key Findings

### 1. The Predictability Ceiling
All algorithms plateau at R² ≈ 0.18 for Kharif crops and 0.34 for Rabi Crops. This is not a model limitation but a **data limitation**—June satellite data lacks information about monsoon rainfall (July-August), which determines final outcomes.
Meanwhile, generated Tiffs for predictions come to be closer than expected to actual Tiff images, due to spatial pattern correlation.

### 2. Spatial Patterns vs Exact Values
The predicted images show correct spatial structure:
- Pattern correlation with actual data: **0.65**
- Pixel-level R²: **0.18**
- Mean absolute error: **0.15**

The model successfully identifies **where** vegetation occurs but cannot predict **exactly how healthy** it will be—a fundamental limitation of satellite-only forecasting.

### 3. Feature Importance
| Rank | Kharif | Importance | Rabi | Importance |
|------|--------|------------|------|------------|
| 1 | jun_B02 (blue) | 0.215 | nov_B02 (blue) | 0.265 |
| 2 | jun_B08 (NIR) | 0.175 | nov_B08 (NIR) | 0.160 |
| 3 | jun_B04 (red) | 0.155 | nov_B04 (red) | 0.150 |

The blue band dominates both seasons—likely a proxy for moisture conditions.

### 4. Seasonal Asymmetry
- **Kharif (monsoon):** R² = 0.18 (low predictability due to rainfall uncertainty)
- **Rabi (winter):** R² = 0.34 (moderate predictability due to irrigation)
- Pattern correlation with actual data: **0.65**

## Interpretation

June satellite data contains information about **where** vegetation will grow (spatial patterns) but insufficient information about **how much** (exact values). This quantifies the fundamental limit of satellite-only crop forecasting and highlights the need for integrated environmental data (rainfall, temperature, soil moisture).

## Tech Stack
Data: Sentinel-2 (ESA Copernicus)

Processing: QGIS, GDAL, rasterio

Analysis: Python, numpy, pandas

ML: scikit-learn, xgboost

Visualization: matplotlib, seaborn
