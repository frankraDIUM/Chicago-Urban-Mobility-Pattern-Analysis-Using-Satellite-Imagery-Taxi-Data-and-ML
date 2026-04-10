# 🚕 Chicago Urban Mobility Pattern Analysis Using Satellite Imagery,Taxi-Data and ML
This project analyzes urban mobility patterns in Chicago by combining satellite imagery (Sentinel-2), OpenStreetMap road data, population density, business licenses/POI data, and a large Chicago taxi dataset (~14 million trips in 2024–early 2026).

---
Dashboard Preview

<p align="center">
  <img src="https://github.com/frankraDIUM/Chicago-Urban-Mobility-Pattern-Analysis-Using-Satellite-Imagery-Taxi-Data-and-ML/blob/main/mobility_analysis_.gif" />
</p>

---




*1. Objectives*

- Detect and quantify vehicle density proxies from Sentinel-2 imagery
- Analyze spatiotemporal mobility trends using taxi data as ground truth
- Identify congestion hotspots and hidden demand areas
- Build predictive models for taxi trip volume (static + temporal)
- Evaluate the added value of satellite-derived features


*2. Data Sources*

- Satellite: Sentinel-2 Level-2A (10m resolution: B02, B03, B04, B08)
- Ground Truth: Chicago Taxi Trips 2024 (Parquet, ~14 million records)
- Road Network: Illinois OSM extract (illinois-260330.osm.pbf)
- Population: WorldPop 2020 (usa_ppp_2020_UNadj.tif)
- POI / Business: Chicago Business Licenses dataset
- Boundaries: Chicago Community Areas (77 areas)

*3. Methodology & Key Components*

**Phase 1: Setup & Exploration**

Environment setup with geospatial stack (GeoPandas, Rasterio, Pyrosm/OSMnx)

Data loading, CRS alignment, clipping Sentinel-2 to Chicago boundary


**Phase 2: Satellite Processing**

- Road mask creation (20m buffer around major OSM roads)
- Vehicle density proxy using texture analysis (GLCM Contrast + NDVI)
- Refined proxy with multi-band features


**Phase 3: Hybrid Analysis**
- Community-area aggregation of taxi trips
- Integration of population density and business/POI counts
- Hotspot analysis using Getis-Ord Gi*

**Phase 4: Modeling & Dashboard**

- Static regression model (R² ≈ 0.666) using hotels, airport flag, distance to Loop, and POI counts
- Temporal model (hourly prediction): R² = 0.954, MAE = 294 trips
- Interactive Streamlit dashboard with maps, residuals, and prediction tool

**Phase 5–6: Advanced Analysis**

- Residual analysis to identify hidden hotspots (O’Hare, Near North Side) and over-predicted areas (far South Side)
- Feature engineering including hotel_airport_interaction, dist_to_ohare_km, centrality, and south_side_flag

*4. Key Findings*

- Destination-Driven Demand
  Taxi trips in Chicago are strongly driven by destinations rather than the residential population. Number of hotels emerged as the strongest predictor, followed by airport presence and proximity to the Loop.
- Limited Value of Sentinel-2 at 10m
  The vehicle density proxy derived from texture analysis added very little predictive power once destination-based features were included. This highlights the resolution limitation of Sentinel-2 for fine-grained vehicle detection in urban environments.
  
- Spatial Patterns
  Strong hotspots in Loop and Near North Side.
  Significant under-prediction at O’Hare Airport.
  Systematic over-prediction in many far South Side community areas.

- Temporal Patterns
  Clear afternoon/evening peak with highest demand at 5 PM (17:00).
  Strong weekday vs weekend differences.


*5. Model Performance*

- Static Community-Area Model: R² ≈ 0.666, MAE ≈ 50,806 trips
- Temporal Hourly Model: R² = 0.954, MAE = 294 trips

*6. Interactive Dashboard*
A Streamlit dashboard was developed featuring:

- Toggleable choropleth maps with custom boxed legends
- Residual analysis map highlighting hidden hotspots
- Temporal prediction tool (select community area + hour + weekend flag)
- Model performance and feature importance visualization

*7. Limitations*

- Sentinel-2 10m resolution is insufficient for accurate individual vehicle counting
- Single-date satellite imagery limits temporal satellite analysis
- Some South Side over-prediction suggests missing socio-economic or transit accessibility features
- POI data relies on business licenses (may not capture all tourist/commercial activity)

*8. Future Work*

- Incorporate higher-resolution imagery (Planet Labs or aerial) for improved vehicle detection
- Add more specific POI types (universities, stadiums, convention centers, tourist attractions)
- Develop true multi-step temporal forecasting using LSTM or Transformer models
- Integrate real-time or multi-date Sentinel data for dynamic congestion monitoring
- Expand the analysis to other major cities for comparative insights
