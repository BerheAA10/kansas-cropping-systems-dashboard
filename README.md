# Kansas Cropping Systems Decision Dashboard

An interactive Streamlit dashboard for comparing Kansas continuous and rotational cropping systems under rainfed, irrigated, and potential-production conditions from 1981–2018.

## Systems

- Continuous maize, sorghum, wheat, and soybean
- Soybean–maize
- Soybean–maize–sorghum
- Soybean–maize–sorghum–wheat

## Dashboard capabilities

- Annual and cumulative crop yield
- Gross return and returns after operating and total production costs
- Applied irrigation and irrigation water-use efficiency
- Producer-oriented system comparisons
- Spatial maps using the 2,776 DSSAT observation sites, with optional filling of internal display gaps

## Data basis

The dashboard uses processed DSSAT simulation results and crop-year economic data for 1981–2018. Economic outputs are expressed per acre.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Required deployment data

The repository must include:

- `data/processed/cropping_systems_long.csv`
- `data/processed/cropping_systems_spatial.parquet`
- `data/economic_returns_1981_2018.csv`

## Streamlit Community Cloud

Deploy `app.py` from the repository root. Select Python 3.12 in Advanced settings.
