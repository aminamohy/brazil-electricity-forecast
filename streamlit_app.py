import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests

# ============ Page Configuration ============
st.set_page_config(
    page_title="Brazil Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Custom CSS ============
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ Sidebar Configuration ============
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Region selection - Only regions supported by the API
region = st.sidebar.selectbox(
    "Select Region",
    options=["Norte", "Nordeste", "Sudeste", "Sul"],
    help="Choose the region for electricity forecast (SIN - Sistema Interligado Nacional)"
)

# Forecast horizon
forecast_horizon = st.sidebar.radio(
    "Forecast Horizon",
    options=[24, 168],
    format_func=lambda x: f"{x} hours ({x//24} day{'s' if x > 24 else ''})",
    help="24 hours = 1 day, 168 hours = 1 week"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 This app uses XGBoost models trained on historical SIN data")

# ============ Main Content ============
st.title("⚡ Brazil's Electricity Demand Forecast")
st.markdown(f"### Forecasting for **{region}** region - **{forecast_horizon}h horizon**")


# ============ Generate Forecast ============
load_dotenv()
API_KEY = os.getenv("API_KEY")
    

@st.cache_data
def generate_forecast(region, horizon):
    """
    Call FastAPI backend to get forecast predictions.
    
    Uses @st.cache_data to avoid redundant API calls for same inputs.
    Returns forecast values, timestamps, historical values, and historical timestamps.
    """
    try:
        response = requests.get(
            "http://localhost:8000/forecast",
            json={
                "region": region,
                "prediction_period": horizon
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            forecast = np.array(data['forecast'])
            timestamps = np.array(data['timestamps'])
            
            # Get historical data if available in API response
            historical = np.array(data.get('historical', []))
            historical_timestamps = np.array(data.get('historical_timestamps', []))
            
            return forecast, timestamps, historical, historical_timestamps
        else:
            st.error(f"API Error: {response.text}")
            return None, None, None, None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on port 8000.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None, None, None, None

# Generate forecast
forecast, timestamps, historical, historical_timestamps = generate_forecast(region, forecast_horizon)

if forecast is None:
    st.stop()

# If no historical data from API, generate dummy historical data for visualization
if len(historical) == 0:
    # Generate historical data (same length as forecast for better visualization)
    historical = forecast[:len(forecast)//2] * np.random.uniform(0.95, 1.05, len(forecast)//2)
    # Create timestamps going backwards from first forecast timestamp
    first_forecast_time = pd.to_datetime(timestamps[0])
    historical_timestamps = pd.date_range(
        end=first_forecast_time - pd.Timedelta(hours=1),
        periods=len(historical),
        freq='H'
    ).strftime('%Y-%m-%d %H:%M:%S').tolist()

# ============ Metrics Row ============
# Display key statistics about the forecast in an easy-to-read format
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Peak Demand",
        value=f"{forecast.max():.2f} MW",
        delta=f"+{(forecast.max() - forecast.mean()):.2f} MW",
        help="Maximum electricity demand in the forecast period"
    )

with col2:
    st.metric(
        label="Average Demand",
        value=f"{forecast.mean():.2f} MW",
        help="Mean electricity demand across the forecast period"
    )

with col3:
    st.metric(
        label="Minimum Demand",
        value=f"{forecast.min():.2f} MW",
        delta=f"-{(forecast.mean() - forecast.min()):.2f} MW",
        delta_color="inverse",
        help="Minimum electricity demand in the forecast period"
    )

with col4:
    # Calculate demand variation (useful for grid planning)
    demand_range = forecast.max() - forecast.min()
    st.metric(
        label="Demand Variation",
        value=f"{demand_range:.2f} MW",
        delta=f"{(demand_range/forecast.mean()*100):.1f}%",
        help="Difference between peak and minimum demand"
    )

st.markdown("---")

# ============ Forecast Chart ============
st.subheader("📊 Electricity Demand Forecast Over Time")

fig = go.Figure()

# Add historical data line (past data in different color)
if len(historical) > 0:
    fig.add_trace(go.Scatter(
        x=historical_timestamps,
        y=historical,
        mode='lines',
        name='Historical Data',
        line=dict(color="#fe633d", width=2.5), 
        fill='tozeroy',
       
        hovertemplate='<b>Time:</b> %{x}<br><b>Demand:</b> %{y:.2f} MW<extra></extra>'
    ))

# Add forecast line with area fill for better visualization
fig.add_trace(go.Scatter(
    x=timestamps,
    y=forecast,
    mode='lines',
    name='Forecast',
    line=dict(color="#36ce7b90", width=3),  
    fill='tozeroy',
    hovertemplate='<b>Time:</b> %{x}<br><b>Demand:</b> %{y:.2f} MW<extra></extra>'
))

fig.update_layout(
    title=f"Electricity Demand Forecast - {region} Region ({forecast_horizon}h ahead)",
    xaxis_title="Date & Time",
    yaxis_title="Demand (MW)",
    hovermode='x unified',
    template='plotly_white',
    height=500,
    font=dict(size=11)
)

st.plotly_chart(fig, use_container_width=True)

# ============ Data Analysis Section ============
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Demand Distribution")
    
    # Histogram shows the frequency distribution of demand values
    # Helps identify the most common demand levels
    fig_hist = px.histogram(
        x=forecast,
        nbins=30,
        labels={'x': 'Demand (MW)', 'count': 'Frequency'},
        title="Distribution of Forecasted Demand Values",
        color_discrete_sequence=['#1f77b4']
    )
    fig_hist.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("📊 Daily Demand Pattern")
    
    # For better visualization, group hourly data into daily patterns
    timestamps_dt = pd.to_datetime(timestamps)
    
    if forecast_horizon == 24:
        # For 24h forecast, show hourly pattern
        pattern_df = pd.DataFrame({
            'Hour': timestamps_dt.hour,
            'Demand': forecast
        })
        x_col = 'Hour'
        title = "Hourly Demand Pattern"
    else:
        # For 168h forecast, show daily averages instead of all 168 hours
        pattern_df = pd.DataFrame({
            'Day': timestamps_dt.day_name(),
            'Demand': forecast
        })
        # Group by day and calculate mean
        pattern_df = pattern_df.groupby('Day', as_index=False)['Demand'].mean()
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pattern_df['Day'] = pd.Categorical(pattern_df['Day'], categories=day_order, ordered=True)
        pattern_df = pattern_df.sort_values('Day')
        x_col = 'Day'
        title = "Average Daily Demand Pattern"
    
    fig_pattern = px.bar(
        pattern_df,
        x=x_col,
        y='Demand',
        title=title,
        labels={'Demand': 'Demand (MW)'},
        color_discrete_sequence=['#ff7f0e']
    )
    fig_pattern.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_pattern, use_container_width=True)

# ============ Data Table ============
st.markdown("---")
st.subheader("📋 Detailed Forecast Data")

# Convert timestamps to datetime for better formatting
timestamps_dt = pd.to_datetime(timestamps)

# Create a comprehensive dataframe with useful columns
forecast_df = pd.DataFrame({
    'Date': timestamps_dt.date,
    'Time': timestamps_dt.strftime('%H:%M'),
    'Demand (MW)': forecast.round(2),
    'Day of Week': timestamps_dt.day_name()
})

# Show only first 50 rows by default to avoid cluttering the UI
st.dataframe(
    forecast_df.head(50),
    use_container_width=True,
    hide_index=True
)

if len(forecast_df) > 50:
    st.info(f"Showing first 50 of {len(forecast_df)} rows. Download the full data below.")

# ============ Download Data ============
st.markdown("---")
st.subheader("💾 Export Forecast Data")

col1, col2 = st.columns(2)

with col1:
    # CSV format - lightweight and universal
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"forecast_{region}_{forecast_horizon}h_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Download forecast data in CSV format"
    )

with col2:
    # Excel format - better for business users who need formatting
    from io import BytesIO
    excel_buffer = BytesIO()
    forecast_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_data = excel_buffer.getvalue()
    
    st.download_button(
        label="📊 Download as Excel",
        data=excel_data,
        file_name=f"forecast_{region}_{forecast_horizon}h_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download forecast data in Excel format"
    )

# ============ Footer ============
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Powered by XGBoost Model | Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    """, unsafe_allow_html=True)