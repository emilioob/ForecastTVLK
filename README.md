# OTA GMV Forecasting Tool

This repo contains a simple Streamlit app for forecasting

Required columns:

- `Month`
- `GMV`

Optional columns:

- `Campaign`
- `Holiday`
- `Promo`
- `Market Event`
- any other text tag columns you want used in explanations

The app returns:

- A 12-month GMV forecast
- A historical vs forecast plot
- Plain-English explanations for spikes and dips
- A model notes section explaining the time-series method and caveats
- An Excel export and text summary export


At least 24 months is recommended for better seasonality.
