import streamlit as st
import os
from PIL import Image

def load_resized_image(path, size=(600, 350)):
    """Load and resize an image to a consistent size for display."""
    with Image.open(path) as img:
        return img.resize(size)

# --- EDA Section ---
st.title("Brazil Electricity Forecast Dashboard")
st.header("Exploratory Data Analysis (EDA)")

st.markdown("---")

# Only show meaningful EDA plots (specify here)
eda_dir = "images/eda"

# Specify the most useful images and their captions
most_useful_eda = [
    "daily_demand_by_region.png",
    "yearly_demand_by_region.png",
    "boxplot_by_region.png",
    "trend_analysis_by_region.png",
]
eda_captions = {
    "daily_demand_by_region.png": "Daily electricity demand patterns across regions.",
    "yearly_demand_by_region.png": "Yearly demand trends for each region, showing seasonal and annual changes.",
    "boxplot_by_region.png": "Distribution and outliers in electricity demand by region.",
    "trend_analysis_by_region.png": "Long-term demand trends for each region, highlighting overall growth or decline.",
}


st.subheader("Key EDA Insights")
for img_name in most_useful_eda:
    img_path = os.path.join(eda_dir, img_name)
    st.image(Image.open(img_path), width=700)
    st.caption(eda_captions.get(img_name, "No caption provided."))
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")


# --- User Input Section ---
st.header("Forecast Visualization")

regions = ["North", "Northeast", "South", "Southeast"]  # Update with actual regions
region = st.selectbox("Select Region", regions)
period_options = ["24h", "168h"]
pred_period = st.radio("Prediction Period", period_options, index=0)
region_mapping = {
    "North": "N",
    "Northeast": "NE",
    "South": "S",
    "Southeast": "SE"
}
st.markdown("---")

# --- Forecast Visualization Section ---

st.subheader(f"Forecasts for {region} ({pred_period})")
forecast_dir = "images/forecast"
forecast_images = [f for f in os.listdir(forecast_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
filtered_forecasts = []
MAE_values = {
    'NBEATS_24':{'N':75,'NE':206,'S':360,'SE':488},
    'NBEATS_168':{'N':212,'NE':332,'S':575,'SE':969},
    'xgboost_24':{'N':63,'NE':146,'S':238,'SE':309},
    'xgboost_168':{'N':100,'NE': 164,'S':377,'SE':213},
    'ARIMA_24':{'N':182,'NE':590,'S':546,'SE':1648},
    'ARIMA_168':{'N':207,'NE':1200,'S':862,'SE':2091},
    'LSTM_24':{'N':248,'NE':658,'S':1146,'SE':2356},
    'LSTM_168':{'N':257,'NE':672,'S':1295,'SE':2940},
}
for img_name in forecast_images:
    namesplit = img_name.split("_")
    if len(namesplit) < 3:
        continue
    model_name = namesplit[0]
    region_name = namesplit[1]
    period_name = namesplit[2].replace("h.png", "")
    if region_name == region_mapping[region] and period_name == pred_period.replace("h", ""):
        filtered_forecasts.append((model_name, region_name, period_name, img_name))

if filtered_forecasts:
    for i in range(0, len(filtered_forecasts), 2):
        cols = st.columns(min(2, len(filtered_forecasts) - i))
        for col_idx, (model_name, region_name, period_name, img_name) in enumerate(filtered_forecasts[i:i+2]):
            img_path = os.path.join(forecast_dir, img_name)
            with cols[col_idx]:
                mae_index = f"{model_name}_{period_name}"
                st.markdown(f"**{model_name} Model with MAE:{MAE_values[mae_index][region_name]}**")
                resized_img = load_resized_image(img_path)
                st.image(resized_img, width=600)
                st.caption(f"Region: {region_name.capitalize()}, Period: {period_name}h")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Chosen model for deployment: XGBoost")
else:
    st.info("No forecast images found for the selected region and period.")


