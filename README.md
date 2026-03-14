# Metro Interstate Traffic Volume — Time Series Forecasting
 
A comprehensive time series forecasting project analyzing and predicting hourly traffic volume on the I-94 Westbound Interstate highway (Minneapolis–St. Paul, MN). This project covers the full pipeline from exploratory data analysis through classical statistical models to gradient-boosted machine learning.
 
## Table of Contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [Blog Post](#blog-post)
- [References](#references)
 
---
 
## Overview
 
This project analyzes hourly traffic volume data from the I-94 Interstate highway (ATR 301 sensor, roughly midway between Minneapolis and St. Paul, MN). The analysis window spans **September 2015 to September 2018** — three full years of data. The goal is to build accurate forecasting models that predict future traffic patterns one hour ahead, which is critical for:
 
- **Traffic Management**: Real-time congestion reduction, adaptive signal control, and ramp metering
- **Urban Planning**: Infrastructure capacity planning and maintenance scheduling
- **Logistics & Navigation**: Delivery route optimization and ETA estimation
- **Environmental Impact**: Emission estimation and air quality forecasting linked to traffic volume
 
### Key Features
 
✅ **Comprehensive EDA** — Distribution analysis, temporal patterns, weather/holiday effects, hourly heatmaps  
✅ **Time Series Analysis** — ADF + KPSS stationarity tests, ACF/PACF, additive seasonal decomposition  
✅ **Feature Engineering** — 42 features: cyclical encoding, lag variables (1–168h), rolling statistics, historical hour×day encoding  
✅ **Multiple Models** — SARIMA (daily), XGBoost (hourly), Gradient Boosting (hourly), Naive seasonal baseline  
✅ **Model Comparison** — MAE, RMSE, MAPE, R² with per-hour breakdown and feature importance  
✅ **Forecasting with Confidence Intervals** — Empirical 50% and 90% prediction intervals  
 
---
 
## Dataset
 
**Source**: [UCI Machine Learning Repository — Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
 
**Description**: Hourly westbound traffic volume for MN DoT ATR station 301, I-94 Interstate, roughly midway between Minneapolis and St. Paul, MN. Includes weather and holiday information recorded at the same timestamp.
 
**Analysis window**: `2015-09-01` → `2018-09-30`
 
**Features**:
 
| Feature | Type | Description |
|---------|------|-------------|
| `holiday` | string / NaN | US national holidays and regional holidays (NaN = normal day) |
| `temp` | float | Average temperature in Kelvin |
| `rain_1h` | float | Amount of rain in the last hour (mm) |
| `snow_1h` | float | Amount of snow in the last hour (mm) |
| `clouds_all` | int | Percentage of cloud cover (0–100) |
| `weather_main` | string | Short weather description (Clear, Rain, Snow, Fog, …) |
| `weather_description` | string | Detailed weather description |
| `date_time` | datetime | Hour of the data in local CST time |
| `traffic_volume` | int | **Target** — hourly I-94 ATR 301 westbound vehicle count |
 
**Key statistics (filtered window)**:
 
| Metric | Value |
|--------|-------|
| Total records | 25,035 hourly observations |
| Missing timestamps | 1,966 (7.3% — sensor gaps) |
| Holiday records | 33 (0.13%) |
| Traffic mean | 3,295 veh/hr |
| Traffic range | 0 – 7,280 veh/hr |
| Temperature range | −29.8°C to +36.9°C |
 
---
 
## Project Structure
 
```
metro-traffic-forecast/
├── metro_traffic_assignment.ipynb   # Main analysis notebook (85 cells)
├── images/                          # Generated figure outputs (fig01–fig19)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore
```
 
> **Note**: The dataset file `Metro_Interstate_Traffic_Volume.csv` is not included in this repository due to size. Download it from the UCI link above and place it in the project root alongside the notebook.
 
---
 
## Installation
 
### 1. Clone the repository
```bash
git clone https://github.com/alinaiman/metro-traffic-forecast.git
cd metro-traffic-forecast
```
 
### 2. Create a virtual environment
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 4. Download the dataset
Download `Metro_Interstate_Traffic_Volume.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and place it in the project root:
```
metro-traffic-forecast/
└── Metro_Interstate_Traffic_Volume.csv   ← place here
```
 
### 5. Launch the notebook
```bash
jupyter notebook metro_traffic_assignment.ipynb
```
Or open in VS Code with the Jupyter extension and select **Run All**.
 
---
 
## Methodology
 
### 1. Exploratory Data Analysis
- Traffic volume distribution (bimodal: overnight zeros + daytime peak ~4,800 veh/hr)
- Temporal patterns by hour, day of week, month, and year
- Weather and holiday effects on traffic volume
- Correlation analysis with external features
- Full 3-year weekly series overview with winter shading
 
### 2. Data Cleaning
- **Temperature**: physically impossible values (< −40°C) replaced by forward-fill
- **Rainfall**: sensor spike at 9,831 mm/hr capped at 300 mm/hr
- **Holiday column**: NaN recoded to binary flag (0 = normal day, 1 = holiday)
 
### 3. Time Series Analysis
- **ADF test**: Augmented Dickey-Fuller — confirmed stationarity (p < 0.001), no differencing required
- **KPSS test**: Confirmed level stationarity
- **Rolling statistics**: 30-day window visual inspection of mean and standard deviation
- **ACF/PACF**: 60-day lag analysis — dominant weekly cycle at lag-7d identified
- **Seasonal decomposition**: Additive model, period = 7 days
 
| Component | Variance Explained |
|-----------|-------------------|
| Trend | 15.5% |
| Seasonal | 62.3% |
| Residual | 20.1% |
 
### 4. Feature Engineering (42 total features)
 
| Category | Features | Count |
|----------|----------|-------|
| Calendar (raw) | hour, dow, month, year, quarter, is_weekend, is_holiday, is_am_rush, is_pm_rush, is_night | 10 |
| Calendar (cyclical) | hour_sin/cos, dow_sin/cos, month_sin/cos | 6 |
| Weather | temp_c, rain_1h, snow_1h, clouds_all, is_rain, is_snow, is_fog | 7 |
| Lag variables | lag_1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | 8 |
| Rolling statistics | rmean & rstd at 3/6/12/24/48/168h windows | 11 |
| Historical encoding | hist_hdow (mean traffic per hour×day-of-week cell) | 1 |
 
> **Cyclical encoding** (sin/cos) is used for periodic features so that hour 23 and hour 0 are correctly treated as adjacent, not 23 units apart.
 
> **Data leakage prevention**: all rolling statistics use `shift(1)` before the rolling window so no future data leaks into the target.
 
### 5. Train / Test Split
- **Train**: Sep 2015 – Aug 2018 (24,147 hourly records)
- **Test**: September 2018 — last 30 days (720 hours), strictly unseen
- Strict chronological order — no shuffling
 
---
 
## Models Implemented
 
| Model | Granularity | Architecture |
|-------|-------------|--------------|
| **SARIMA(1,0,1)(1,1,0)[7]** | Daily | Non-seasonal: AR(1)+MA(1); Seasonal: AR(1)+1 diff, period=7 |
| **XGBoost** | Hourly | 500 trees, max_depth=6, lr=0.05, subsample=0.8 |
| **Gradient Boosting (GBM)** | Hourly | 400 trees, max_depth=5, lr=0.07, min_samples_leaf=20 |
| **Naive seasonal baseline** | Hourly | Prediction = traffic at same weekday+hour 1 week ago (lag_168h) |
 
---
 
## Results
 
### Model Performance — Test Set (September 2018, 720 hours)
 
| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|-------|-------|--------|--------|------|
| **XGBoost** | **140.5** | **224.2** | **5.73%** | **0.9868** |
| GBM | 144.3 | 228.2 | 6.07% | 0.9863 |
| SARIMA (daily) | 175.7 | 302.4 | 6.57% | — |
| Naive (lag-168h) | 282.2 | 592.4 | 12.29% | 0.9079 |
 
**XGBoost achieves ~50% lower MAE than the seasonal naive baseline.**
 
### Prediction Interval Coverage (XGBoost, empirical)
 
| Interval | Nominal | Actual Coverage |
|----------|---------|-----------------|
| 90% PI | 90% | 84.2% |
| 50% PI | 50% | 44.0% |
 
### Top 10 Features — XGBoost
 
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `hist_hdow` | 0.5123 |
| 2 | `hour_cos` | 0.2400 |
| 3 | `is_weekend` | 0.0963 |
| 4 | `lag_1` | 0.0343 |
| 5 | `rmean_3` | 0.0234 |
| 6 | `hour` | 0.0204 |
| 7 | `is_night` | 0.0202 |
| 8 | `rstd_12` | 0.0060 |
| 9 | `hour_sin` | 0.0049 |
| 10 | `lag_24` | 0.0029 |
 
### Hardest vs. Easiest Hours to Predict (XGBoost MAE)
 
| Hardest | MAE | Easiest | MAE |
|---------|-----|---------|-----|
| 23:00 | 326 | 03:00 | 22 |
| 07:00 | 251 | 02:00 | 39 |
| 21:00 | 236 | 04:00 | 48 |
| 17:00 | 194 | 01:00 | 57 |
| 22:00 | 193 | 00:00 | 84 |
 
---
 
## Key Findings
 
1. **Historical context dominates**: `hist_hdow` alone accounts for 51% of XGBoost's feature importance — knowing the average traffic for "Tuesday at 8 AM" is more predictive than any weather signal.
 
2. **Strong dual seasonality**: clear 24-hour intraday cycle (AM peak 7–8h, PM peak 16–17h) and a 7-day weekly cycle (weekday avg 3,500+ veh/hr vs. weekend avg ~1,900 veh/hr).
 
3. **Holiday effect is dramatic**: only 33 holiday records in the analysis window, yet they produce a −72% drop in traffic compared to normal days.
 
4. **Series is stationary at level**: ADF p-value < 0.001, KPSS confirms — no differencing required. SARIMA can model the series without transformation.
 
5. **ML outperforms SARIMA on hourly data**: SARIMA on daily aggregates achieves MAPE 6.57%; XGBoost on hourly data with engineered features achieves 5.73% and captures non-linear patterns SARIMA cannot model.
 
6. **Weather has weak direct effect**: temperature correlation with traffic is only r = 0.13. Time-of-day and day-of-week carry almost all the signal.
 
7. **Transition hours are hardest to predict**: 23:00 (late-night drop-off), 7:00 AM (morning ramp-up), and 17:00 (evening peak) have the highest MAE — these are moments of rapid change where small timing shifts cause large errors.
 
---
 
## Usage
 
Open `metro_traffic_assignment.ipynb` in Jupyter or VS Code, ensure `Metro_Interstate_Traffic_Volume.csv` is in the same folder, and run all cells top-to-bottom. The notebook is fully self-contained — it filters the data, engineers features, trains models, and saves all 19 figures automatically.
 
```python
# The notebook filters to the analysis window automatically:
START, END = '2015-09-01', '2018-09-30'
df = df_raw[(df_raw['date_time'] >= START) & (df_raw['date_time'] <= END)]
```
 
Figures are saved as `images/fig01_outliers.png` through `images/fig19_residuals.png`.
 
---
 
## Blog Post
 
**Medium**: [Predicting Rush Hour: How Machine Learning Forecasts Highway Traffic 50% Better Than Common Sense](your-medium-link-here)
 
The post covers: dataset introduction, EDA insights, methodology comparison (SARIMA vs. XGBoost), model results, real-world applications, and lessons learned.
 
---
 
## References
 
1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD '16*, 785–794.
3. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics, 29*(5), 1189–1232.
5. UCI Machine Learning Repository: [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
 
---
 
## Acknowledgments
 
- Dataset provided by the UCI Machine Learning Repository
- Minnesota Department of Transportation (MnDOT) for ATR 301 sensor data
 
---
 
**Author**: Alina Imanakhunova  
**Course**: Time Series Analysis  
**Date**: March 2026

