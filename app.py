import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
import numpy as np
import joblib
import xgboost as xgb

st.set_page_config(page_title="Chicago Mobility Dashboard", layout="wide")
st.title("🚕 Chicago Urban Mobility Analysis")

@st.cache_data
def load_data():
    comm = gpd.read_file('chicago_community_areas_with_residuals.geojson')
    for col in comm.columns:
        if pd.api.types.is_datetime64_any_dtype(comm[col]):
            comm[col] = comm[col].astype(str)
    data_df = comm.drop(columns=['geometry']).copy()
    return comm, data_df

comm_areas, data_df = load_data()
geo_json = json.loads(comm_areas.to_json())

# Load temporal model
@st.cache_resource
def load_temporal_model():
    try:
        return joblib.load('taxi_temporal_model_native.pkl')
    except:
        return None

temporal_model = load_temporal_model()

# Add legend
def add_custom_legend(m, title, bins, colors):
    legend_html = f'''
     <div style="position: fixed;
     bottom: 50px; left: 50px; width: auto; min-width: 180px; height: auto;
     background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
     padding: 12px; color: black; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); border-radius: 5px;">
     <b style="display: block; margin-bottom: 10px; font-size: 14px; border-bottom: 1px solid #ccc; padding-bottom: 5px; color: black;">{title}</b>
     <table style="border-spacing: 0 5px; border-collapse: separate;">
    '''
    for i in range(len(colors)):
        start = f"{int(bins[i]):,}"
        end = f"{int(bins[i+1]):,}"
        legend_html += f'''
        <tr>
            <td style="vertical-align: middle;">
                <div style="background:{colors[i]}; width:20px; height:20px; border:1px solid black;"></div>
            </td>
            <td style="vertical-align: middle; padding-left: 10px; white-space: nowrap; font-weight: 500; color: black;">
                {start} &ndash; {end}
            </td>
        </tr>
        '''
    legend_html += '</table></div>'
    m.get_root().html.add_child(folium.Element(legend_html))

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Interactive Map", "📊 Model Performance", "🔍 Residuals", "⏰ Temporal Predictions", "📈 Insights"])

# ===================== TAB 1: Interactive Map =====================
with tab1:
    st.header("Interactive Map")
    col_select, _ = st.columns([1, 3])
    with col_select:
        layer = st.selectbox("Select Layer",
                             ["Total Taxi Trips", "Trips per 1,000 People",
                              "Satellite Proxy", "Residuals"])

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="CartoDB positron")

    if layer == "Satellite Proxy":
        folium.Choropleth(
            geo_data=geo_json, data=data_df,
            columns=['pickup_community_area', 'avg_satellite_proxy'],
            key_on='feature.properties.pickup_community_area',
            fill_color='YlOrRd', fill_opacity=0.75, line_opacity=0.3,
            legend_name='Satellite Proxy'
        ).add_to(m)

    else:
        if layer == "Total Taxi Trips":
            col_target, title = 'total_trips', "Total Taxi Trips"
            colors = ['#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
            bins = np.linspace(data_df[col_target].min(), data_df[col_target].max() * 1.0001, 7)

        elif layer == "Trips per 1,000 People":
            col_target, title = 'trips_per_1000_people', "Trips / 1k People"
            colors = ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#756bb1', '#54278f']
            bins = np.linspace(data_df[col_target].min(), data_df[col_target].max() * 1.0001, 7)

        else: # Residuals
            col_target, title = 'residual', "Residuals"
            colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
            max_val = data_df[col_target].abs().max()
            bins = np.linspace(-max_val, max_val * 1.0001, 6)

        def get_color(val):
            if val is None or np.isnan(val): return '#808080'
            for i in range(len(bins)-1):
                if bins[i] <= val < bins[i+1]:
                    return colors[i]
            return colors[-1]

        folium.GeoJson(
            comm_areas,
            style_function=lambda x: {
                'fillColor': get_color(x['properties'].get(col_target, 0)),
                'color': 'black', 'weight': 0.5, 'fillOpacity': 0.75
            },
            tooltip=folium.GeoJsonTooltip(fields=['pickup_community_area', col_target])
        ).add_to(m)

        add_custom_legend(m, title, bins, colors)

    st_folium(m, use_container_width=True, height=800, key="map_main")

# ===================== TAB 2: Model Performance =====================
with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1: st.metric("R² Score", "0.666")
    with col2: st.metric("Mean Absolute Error", "50,806 trips")
    imp_data = pd.DataFrame({'Feature': ['num_hotels', 'is_airport', 'num_bars', 'dist_to_loop_km', 'num_restaurants'], 'Importance': [0.487, 0.217, 0.062, 0.082, 0.055]})
    st.plotly_chart(px.bar(imp_data, x='Importance', y='Feature', orientation='h', title="Top Feature Importance"), use_container_width=True)

# ===================== TAB 3: Residuals =====================
with tab3:
    st.header("Residual Analysis")
    st.write("**Red** = Model under-predicts (Hidden Hotspots)")
    st.write("**Blue** = Model over-predicts")
    st.info("O’Hare and Near North Side are significantly under-predicted.")

    m_res = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="CartoDB positron")
    col_target = 'residual'
    title = "Residuals"
    colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
    max_val = data_df[col_target].abs().max()
    bins_res = np.linspace(-max_val, max_val * 1.0001, 6)

    def get_res_color(val):
        if val is None or np.isnan(val): return '#808080'
        for i in range(len(bins_res)-1):
            if bins_res[i] <= val < bins_res[i+1]:
                return colors[i]
        return colors[-1]

    folium.GeoJson(
        comm_areas,
        style_function=lambda x: {
            'fillColor': get_res_color(x['properties'].get(col_target, 0)),
            'color': 'black', 'weight': 0.5, 'fillOpacity': 0.75
        },
        tooltip=folium.GeoJsonTooltip(fields=['pickup_community_area', col_target])
    ).add_to(m_res)

    add_custom_legend(m_res, title, bins_res, colors)
    st_folium(m_res, use_container_width=True, height=800, key="map_residuals_final")

# ===================== TAB 4: Temporal Predictions =====================
with tab4:
    st.header("⏰ Temporal Predictions")
    if temporal_model is None:
        st.error("Temporal model file not found.")
    else:
        # Use community name if available, otherwise community area number
        area_col = 'community' if 'community' in comm_areas.columns else 'pickup_community_area'
        area_list = sorted(comm_areas[area_col].unique())

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_area = st.selectbox("Community Area", area_list)
        with c2:
            sel_hour = st.slider("Hour of Day", 0, 23, 17)
        with c3:
            is_wknd = st.checkbox("Weekend?")

        # Prepare features
        area_id = comm_areas[comm_areas[area_col] == sel_area]['pickup_community_area'].iloc[0]
        input_df = pd.DataFrame({
            'pickup_community_area': [area_id],
            'hour': [sel_hour],
            'is_weekend': [1 if is_wknd else 0],
            'hour_sin': [np.sin(2 * np.pi * sel_hour / 24)],
            'hour_cos': [np.cos(2 * np.pi * sel_hour / 24)]
        })

        try:

            dinput = xgb.DMatrix(input_df)
            pred = temporal_model.predict(dinput)[0]
            st.metric(f"Predicted Trips for {sel_area}", f"{int(max(0, pred)):,}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===================== TAB 5: Insights =====================
with tab5:
    st.header("Key Insights")
    st.markdown("- **Hotels** are the strongest predictor.\n- **O’Hare Airport** has a massive effect.\n- Some **South Side** areas are over-predicted.")

st.sidebar.success("Dashboard loaded successfully!")
