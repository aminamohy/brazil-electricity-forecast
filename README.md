# Brazil Electricity Load Forecasting

## Project Overview
This repository focuses on forecasting electricity load for **Brazil’s National Interconnected System (SIN)**. Accurate demand forecasting is crucial for the Brazilian energy grid—one of the largest and most complex hydro-thermal networks in the world. It helps avoid blackouts, optimize energy dispatch, and reduce operational costs.  

The repository contains **XGBoost, ARIMA, LSTM, and N-BEATS models** trained and validated on historical electricity load data. Forecasts are provided for both short-term (24-hour horizon) and medium-term (168-hour horizon) predictions.

---

## Business Problem
The **National System Operator (ONS)** faces multiple challenges in managing the grid:

- **Under-Forecasting**: Causes immediate instability, requiring expensive thermal dispatch or risking load shedding (blackouts).  
- **Over-Forecasting**: Leads to wasted generation potential (spilled water in hydro reservoirs) or unnecessary purchases of energy reserves.  
- **Regional Disparity**: Weather events affect regions differently. For instance, a heatwave in the North requires a completely different dispatch strategy than a cold front in the South. A generic national model is insufficient.

---

## Project Objective
Develop a **Regional Time-Series Forecasting Engine** that predicts hourly electricity demand (`val_cargaenergiahomwmed`) for the next **24 to 168 hours**, providing **granular visibility** into each of Brazil's four major power subsystems:

- **SE/CO (Southeast/Central-West)**  
- **S (South)**  
- **N (North)**  
- **NE (Northeast)**  

This enables operators to optimize dispatch planning and improve grid stability.

---

## Dataset
The dataset contains **hourly electricity demand** for the four main regions of Brazil. It is publicly available at:  

[Hourly Electricity Demand Brazil Dataset - Hugging Face](https://huggingface.co/datasets/SamuelM0422/Hourly-Electricity-Demand-Brazil)

**Key features:**

- Hourly demand values (`val_cargaenergiahomwmed`)  
- Regional classification: SE/CO, S, N, NE  
- Time range: [specify the dataset period]  
- Additional features: [weather, holidays, etc., if any]  

---

## Models Used

### XGBoost
- Chosen for **superior performance** compared to LSTM and N-BEATS for this dataset.  
- XGBoost is used for forecasting Brazil’s electricity load because it handles nonlinear patterns and complex interactions in demand data extremely well.
 It also performs strongly with tabular time-series features such as lags, rolling statistics, and weather or calendar variables, making it highly accurate and efficient.

### ARIMA
- Traditional time-series forecasting model.  
- Hyperparameters used: `p=2, d=1, q=0`.  
- Useful for capturing linear trends and seasonality.  

### N-BEATS
- Deep learning time-series model.  
- Hyperparameters optimized using **Optuna**.  
- Designed for capturing complex non-linear patterns in electricity load.  

### LSTM
- Long Short-Term Memory networks for sequential data.  
- Captures temporal dependencies over time in electricity demand.  

---

## Model Performance by Region
- The **best accuracy** was achieved in **Region N** (North), where load patterns were more stable.  
- The **SE region** (Southeast) showed the **highest error** due to larger load variability.  

| Model      | Region | Forecast Horizon | Notes |
|------------|--------|----------------|-------|
| XGBoost    | All    | 24 & 168 hrs   | Best overall performance |
| ARIMA      | All    | 24 & 168 hrs   | p=2, d=1, q=0 |
| N-BEATS    | All    | 24 & 168 hrs   | Hyperparameters tuned with Optuna |
| LSTM       | All    | 24 & 168 hrs   | Captures temporal dependencies |

---

## Note
- Forecast accuracy may vary across regions due to differences in load variability and external factors like weather or holidays.  
- The models were trained on historical load data; future structural changes in the grid or generation mix may affect predictions.  
- Users can retrain or fine-tune models with updated data to maintain performance.  
- All scripts and models assume proper preprocessing of the dataset (handling missing values, scaling, and splitting by region).  
- After comparing multiple models including ARIMA, LSTM, and N-BEATS, **XGBoost consistently achieved the best performance** across both short-term (24-hour) and medium-term (168-hour) forecasts. It provided the most accurate predictions, particularly in regions with stable load patterns, making it the recommended model for electricity load forecasting in Brazil’s SIN.
 
---

## References
- [ONS - Operador Nacional do Sistema Elétrico](https://www.ons.org.br/)  
- [Hourly Electricity Demand Brazil Dataset - Hugging Face](https://huggingface.co/datasets/SamuelM0422/Hourly-Electricity-Demand-Brazil)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)  
- [ARIMA Time-Series Modeling - Statsmodels](https://www.statsmodels.org/stable/tsa.html)  

