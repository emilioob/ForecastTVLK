# OTA GMV Forecasting Tool

This repo contains a simple Streamlit app for forecasting OTA GMV.

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

## Run locally

```powershell
cd "C:\Users\Emilio\Documents\Codex\TVLK Financial Forecasting Tool"
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

You can also double-click `Launch TVLK Forecasting Tool.bat` from the project folder.

## Build Windows package

```powershell
cd "C:\Users\Emilio\Documents\Codex\TVLK Financial Forecasting Tool"
.\build_windows_package.ps1
```

This keeps the app on Streamlit, builds a PyInstaller `onedir` package, and creates a user-ready zip in `.\release`.

## Input format

Use one row per month.

Example:

```csv
Month,GMV,Holiday,Campaign
2025-01-01,1230000,New Year,Winter Push
2025-02-01,1150000,,Winter Push
```

At least 24 months is recommended for better seasonality.
