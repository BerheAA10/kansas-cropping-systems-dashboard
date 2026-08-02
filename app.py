from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_economic_data,
    load_simulation_data,
    load_spatial_simulation_data,
)
from src.metrics import (
    OBJECTIVES,
    add_crop_cumulative_metrics,
    add_economic_and_water_metrics,
    aggregate_crop_year,
    aggregate_system_year,
    rank_systems,
    summarize_systems,
)
from src.spatial import (
    build_complete_master_grid,
    compact_money,
    idw_fill_complete_grid,
    robust_color_range,
)

ROOT = Path(__file__).resolve().parent
SIM_PATH = ROOT / "data" / "processed" / "cropping_systems_long.csv"
ECON_PATH = ROOT / "data" / "economic_returns_1981_2018.csv"
SPATIAL_PATH = ROOT / "data" / "processed" / "cropping_systems_spatial.parquet"
MASTER_GRID_PATH = ROOT / "data" / "processed" / "master_site_grid.csv"

FARM_ECONOMIC_OPTIONS = {
    "Gross return": {
        "annual": "gross_income_simulated_usd_ac",
        "mean": "mean_gross_income_usd_ac",
        "total": "total_gross_income_usd_ac",
        "definition": "Simulated yield × annual crop price.",
    },
    "Return after operational cost": {
        "annual": "return_above_operating_usd_ac",
        "mean": "mean_return_above_operating_usd_ac",
        "total": "total_return_above_operating_usd_ac",
        "definition": "Gross return minus annual operational cost.",
    },
    "Return after total production cost": {
        "annual": "return_above_total_usd_ac",
        "mean": "mean_return_above_total_usd_ac",
        "total": "total_return_above_total_usd_ac",
        "definition": "Gross return minus annual total production cost.",
    },
    "Marginal gross return vs rainfed": {
        "annual": "marginal_gross_return_usd_ac",
        "mean": "mean_marginal_gross_return_usd_ac",
        "total": "total_marginal_gross_return_usd_ac",
        "definition": "Gross-return difference from the matching rainfed crop–system–year.",
    },
    "Marginal return after operational cost vs rainfed": {
        "annual": "marginal_return_above_operating_usd_ac",
        "mean": "mean_marginal_return_above_operating_usd_ac",
        "total": "total_marginal_return_above_operating_usd_ac",
        "definition": "Difference from the matching rainfed crop–system–year after operational cost.",
    },
    "Marginal return after total cost vs rainfed": {
        "annual": "marginal_return_above_total_usd_ac",
        "mean": "mean_marginal_return_above_total_usd_ac",
        "total": "total_marginal_return_above_total_usd_ac",
        "definition": "Difference from the matching rainfed crop–system–year after total production cost.",
    },
}

SPATIAL_MEASURES = {
    "Simulated yield": ("yield_kg_ha", "Yield (kg/ha)", False),
    "Gross return": ("gross_income_simulated_usd_ac", "Gross return ($/acre)", True),
    "Return after operational cost": ("return_above_operating_usd_ac", "Return after operational cost ($/acre)", True),
    "Return after total production cost": ("return_above_total_usd_ac", "Return after total production cost ($/acre)", True),
    "Marginal gross return vs rainfed": ("marginal_gross_return_usd_ac", "Marginal gross return ($/acre)", True),
    "Marginal return after operational cost vs rainfed": ("marginal_return_above_operating_usd_ac", "Marginal operating return ($/acre)", True),
    "Applied irrigation": ("irrigation_mm", "Applied irrigation (mm)", False),
    "Incremental IWUE": ("incremental_iwue_kg_m3", "Incremental IWUE (kg/m³)", False),
}

SPATIAL_COLOR_STYLES = {
    "Sampled texture: green–yellow–red": {
        # low values = red, middle = yellow/cream, high values = green
        "scale": [
            [0.00, "#c81d25"],
            [0.16, "#dc5f4b"],
            [0.35, "#efb285"],
            [0.50, "#efecc8"],
            [0.68, "#c9dfad"],
            [0.84, "#73c05f"],
            [1.00, "#1a9850"],
        ],
        "line_color": "rgba(255,255,255,0.58)",
        "background": "#dcdcdc",
    },
    "Green–yellow–orange–red stepped classes": {
        # stepped-class version with low values in red and high values in green
        "scale": [
            [0.00, "#c81d25"],
            [0.1999, "#c81d25"],
            [0.20, "#e67e5f"],
            [0.3999, "#e67e5f"],
            [0.40, "#f1d9a6"],
            [0.5999, "#f1d9a6"],
            [0.60, "#b8d989"],
            [0.7999, "#b8d989"],
            [0.80, "#74c25b"],
            [1.00, "#1a9850"],
        ],
        "line_color": "rgba(255,255,255,0.62)",
        "background": "#dcdcdc",
    },
    "Purple–gold": {
        "scale": [
            [0.00, "#05010a"],
            [0.18, "#2b0a6d"],
            [0.40, "#7b239a"],
            [0.62, "#cf4b7c"],
            [0.82, "#f59d62"],
            [1.00, "#efe9b5"],
        ],
        "line_color": "rgba(255,255,255,0.34)",
        "background": "#efefef",
    },
}

EXPECTED_CROPPING_SYSTEMS = [
    "MZ",
    "SG",
    "WT",
    "SB",
    "SB-MZ",
    "SB-MZ-SG",
    "SB-MZ-SG-WT",
]
EXPECTED_WATER_REGIMES = ["Rainfed", "Irrigated", "Potential"]
SYSTEM_CROPS = {
    "MZ": {"MZ"},
    "SG": {"SG"},
    "WT": {"WT"},
    "SB": {"SB"},
    "SB-MZ": {"SB", "MZ"},
    "SB-MZ-SG": {"SB", "MZ", "SG"},
    "SB-MZ-SG-WT": {"SB", "MZ", "SG", "WT"},
}
CROP_ORDER = ["Soybean", "Maize", "Sorghum", "Wheat"]
CROP_COLOR_MAP = {
    "Soybean": "#2ca02c",
    "Maize": "#ff7f0e",
    "Sorghum": "#9467bd",
    "Wheat": "#1f77b4",
}
SYSTEM_COLOR_MAP = {
    "MZ": "#e67e22",
    "SG": "#8e44ad",
    "WT": "#2980b9",
    "SB": "#27ae60",
    "SB-MZ": "#16a085",
    "SB-MZ-SG": "#c0392b",
    "SB-MZ-SG-WT": "#34495e",
}
WATER_DASH_MAP = {"Rainfed": "solid", "Irrigated": "dash", "Potential": "dot"}
WATER_SYMBOL_MAP = {"Rainfed": "circle", "Irrigated": "square", "Potential": "diamond"}

st.set_page_config(
    page_title="Kansas Cropping Systems Decision Dashboard",
    page_icon="🌾",
    layout="wide",
)


def sorted_available(values: pd.Series, expected: list[str]) -> list[str]:
    found = set(values.dropna().astype(str))
    return [value for value in expected if value in found]


def crop_chart(
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    all_systems_selected: bool,
    markers: bool = True,
    money_axis: bool = False,
):
    # Plotly raises a ValueError when a selected Farm economics field is not
    # present. Return None instead, allowing the interface to explain why the
    # measure is unavailable without terminating the whole app.
    if y_col not in data.columns:
        return None
    chart_data = data.dropna(subset=[y_col]).copy()
    if chart_data.empty:
        return None
    category_orders = {
        "crop_name": CROP_ORDER,
        "water_regime": EXPECTED_WATER_REGIMES,
        "base_system": EXPECTED_CROPPING_SYSTEMS,
    }
    common = dict(
        data_frame=chart_data,
        x="year",
        y=y_col,
        color="crop_name",
        line_dash="water_regime",
        symbol="water_regime",
        markers=markers,
        color_discrete_map=CROP_COLOR_MAP,
        line_dash_map=WATER_DASH_MAP,
        symbol_map=WATER_SYMBOL_MAP,
        category_orders=category_orders,
        labels={
            "year": "Year",
            y_col: y_label,
            "crop_name": "Crop",
            "water_regime": "Water regime",
            "base_system": "Cropping system",
        },
        title=title,
    )
    if all_systems_selected:
        fig = px.line(
            **common,
            facet_col="base_system",
            facet_col_wrap=2,
        )
        fig.for_each_annotation(
            lambda annotation: annotation.update(
                text=annotation.text.replace("base_system=", "")
            )
        )
        fig.update_yaxes(matches=None)
        fig.update_layout(height=max(580, 290 * int(np.ceil(chart_data["base_system"].nunique() / 2))))
    else:
        fig = px.line(**common)
        fig.update_layout(height=500)
    fig.update_layout(legend_title_text="Crop / water regime")
    fig.update_traces(marker={"size": 7}, line={"width": 2.2})
    fig.update_yaxes(tickformat="~s", tickprefix="$" if money_axis else "")
    return fig


def style_numeric_axis(fig, *, money: bool = False, axis: str = "y"):
    update = {"tickformat": "~s"}
    if money:
        update["tickprefix"] = "$"
    if axis == "x":
        fig.update_xaxes(**update)
    else:
        fig.update_yaxes(**update)
    return fig


def reset_optimizer_state():
    for key in [
        "optimizer_use_water_limit",
        "optimizer_use_return_floor",
        "optimizer_complete_only",
        "optimizer_water_limit",
        "optimizer_return_floor",
        "optimizer_regime",
        "optimizer_objective",
    ]:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def cached_load_spatial(path: str) -> pd.DataFrame:
    return load_spatial_simulation_data(path)


st.title("Kansas Cropping Systems: Yield, Income, and Water-Use Dashboard")

economics = load_economic_data(ECON_PATH)

if not SIM_PATH.exists():
    st.warning(
        "The economic data are available, but the processed simulation dataset has not been created."
    )
    st.code('& ".\\prepare_H_drive_data.ps1"', language="powershell")
    st.stop()

try:
    simulation = load_simulation_data(SIM_PATH)
except Exception as exc:
    st.error(f"Could not load the simulation dataset: {exc}")
    st.stop()

available_systems = sorted_available(simulation["base_system"], EXPECTED_CROPPING_SYSTEMS)
available_regimes = sorted_available(simulation["water_regime"], EXPECTED_WATER_REGIMES)
spatial_available = SPATIAL_PATH.exists()
missing_expected_systems = [
    system for system in EXPECTED_CROPPING_SYSTEMS if system not in available_systems
]

system_choices = ["All cropping systems", *available_systems]
default_system_index = 1 if len(system_choices) > 1 else 0
regime_choices = ["All water regimes", *available_regimes]

with st.sidebar:
    st.subheader("Water regime")
    selected_regime_label = st.selectbox(
        "Water regime",
        regime_choices,
        index=0,
        label_visibility="collapsed",
        help="Choose one regime or compare all available regimes.",
    )

    st.subheader("Cropping system")
    selected_system_label = st.selectbox(
        "Cropping system",
        system_choices,
        index=default_system_index,
        label_visibility="collapsed",
        help="One system is shown at a time by default. Select All cropping systems for comparison.",
    )

    st.subheader("Farm economics")
    selected_economic_label = st.selectbox(
        "Farm economics",
        list(FARM_ECONOMIC_OPTIONS),
        index=1,
        label_visibility="collapsed",
    )

    st.subheader("Spatial farm options")
    with st.expander("Map filters", expanded=True):
        spatial_system = st.selectbox(
            "Map cropping system",
            available_systems,
            index=(available_systems.index(selected_system_label) if selected_system_label in available_systems else 0),
            disabled=not spatial_available,
        )

        system_regimes = set(
            simulation.loc[simulation["base_system"].eq(spatial_system), "water_regime"]
            .dropna()
            .astype(str)
        )
        spatial_regime_options = [
            regime for regime in ["Rainfed", "Irrigated"] if regime in system_regimes
        ]
        spatial_regime = st.selectbox(
            "Map water regime",
            spatial_regime_options or ["No rainfed or irrigated data"],
            disabled=not spatial_available or not spatial_regime_options,
        )

        available_map_years = sorted(
            int(year)
            for year in simulation.loc[
                simulation["base_system"].eq(spatial_system)
                & simulation["water_regime"].eq(spatial_regime),
                "year",
            ].dropna().unique()
        )
        spatial_year = st.select_slider(
            "Map year",
            options=available_map_years or [1981],
            value=(available_map_years[-1] if available_map_years else 1981),
            disabled=not spatial_available or not available_map_years,
        )
        default_spatial_measure = (
            selected_economic_label
            if selected_economic_label in SPATIAL_MEASURES
            else "Return after operational cost"
        )
        spatial_measure = st.selectbox(
            "Map measure",
            list(SPATIAL_MEASURES),
            index=list(SPATIAL_MEASURES).index(default_spatial_measure),
            disabled=not spatial_available or not available_map_years,
        )
        spatial_style_options = list(SPATIAL_COLOR_STYLES)
        spatial_style = st.selectbox(
            "Map style",
            spatial_style_options,
            index=spatial_style_options.index("Sampled texture: green–yellow–red"),
            disabled=not spatial_available or not available_map_years,
            help="Choose a color and texture style similar to the example maps.",
        )
        if spatial_available and available_map_years:
            st.caption("The selected map is displayed automatically in the Spatial farm maps tab.")
        else:
            st.caption(
                "Rerun prepare_H_drive_data.ps1 to create the spatial dataset, or select a system with rainfed/irrigated records."
            )
    spatial_enabled = spatial_available and bool(spatial_regime_options) and bool(available_map_years)

selected_systems = (
    available_systems
    if selected_system_label == "All cropping systems"
    else [selected_system_label]
)
selected_regimes = (
    available_regimes
    if selected_regime_label == "All water regimes"
    else [selected_regime_label]
)
all_systems_selected = selected_system_label == "All cropping systems"

selected_economic = FARM_ECONOMIC_OPTIONS[selected_economic_label]
selected_annual_economic_col = selected_economic["annual"]
selected_mean_economic_col = selected_economic["mean"]
selected_total_economic_col = selected_economic["total"]

if missing_expected_systems:
    st.warning(
        "Not yet present in the processed CSV: "
        + ", ".join(missing_expected_systems)
        + ". Rebuild the processed data after installing the newest preparation script."
    )

metric_source = simulation[simulation["base_system"].isin(selected_systems)].copy()

# Retain all regimes for matching rainfed baselines before applying the visible regime filter.
detailed_all_regimes = add_economic_and_water_metrics(metric_source, economics)
detailed = detailed_all_regimes[
    detailed_all_regimes["water_regime"].isin(selected_regimes)
].copy()

if detailed.empty:
    st.info("No records match the selected cropping system and water regime.")
    st.stop()

missing_economic_rows = detailed[
    detailed[["price_usd_bu", "operating_cost_usd_ac", "total_production_cost_usd_ac"]]
    .isna()
    .any(axis=1)
]
if not missing_economic_rows.empty:
    missing_pairs = (
        missing_economic_rows[["crop_code", "year"]]
        .drop_duplicates()
        .sort_values(["crop_code", "year"])
    )
    st.error(
        f"Economic price or cost data are missing for {len(missing_pairs)} crop-year combinations. "
        "Economic charts exclude or show gaps for those records."
    )

annual = aggregate_system_year(detailed)
summary = summarize_systems(annual)
crop_annual = aggregate_crop_year(detailed)
crop_cumulative = add_crop_cumulative_metrics(crop_annual, 1981, 2018)

# The producer optimizer intentionally uses every available system and regime,
# independent of the one-system display selected in the sidebar.
optimizer_detailed_all = add_economic_and_water_metrics(simulation, economics)
optimizer_annual_all = aggregate_system_year(optimizer_detailed_all)
optimizer_summary_all = summarize_systems(optimizer_annual_all)

# Warn about coverage for the currently displayed system(s).
coverage_check = (
    simulation[simulation["base_system"].isin(selected_systems)]
    .groupby(["base_system", "water_regime"], as_index=False)
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        years=("year", "nunique"),
        crops=("crop_code", lambda values: ",".join(sorted(set(values)))),
    )
)
coverage_problems: list[str] = []
for row in coverage_check.itertuples(index=False):
    found_crops = set(str(row.crops).split(",")) if row.crops else set()
    missing_crops = SYSTEM_CROPS.get(row.base_system, set()) - found_crops
    details: list[str] = []
    if row.years < 38:
        details.append(f"{row.years}/38 years")
    if missing_crops:
        details.append("missing crops " + ", ".join(sorted(missing_crops)))
    if details:
        coverage_problems.append(
            f"{row.base_system} | {row.water_regime}: " + "; ".join(details)
        )
if coverage_problems:
    with st.expander("Data coverage notice", expanded=False):
        for problem in coverage_problems:
            st.write("• " + problem)

tabs = st.tabs(
    [
        "Overview",
        "Spatial farm maps",
        "Annual charts",
        "Cumulative performance",
        "Economics",
        "Water use and IWUE",
        "Producer optimizer",
        "Data quality and methods",
    ]
)

with tabs[0]:
    st.subheader(f"Long-term {selected_economic_label.lower()}")
    overview_trend = annual.dropna(subset=[selected_annual_economic_col]).copy()
    if overview_trend.empty:
        st.info(
            "This economic measure is unavailable for the current selection. "
            "Marginal measures require a matching rainfed record for the same cropping system, crop, and year."
        )
    else:
        overview_trend_fig = px.line(
            overview_trend,
            x="year",
            y=selected_annual_economic_col,
            color="base_system",
            line_dash="water_regime",
            symbol="water_regime",
            markers=True,
            color_discrete_map=SYSTEM_COLOR_MAP,
            line_dash_map=WATER_DASH_MAP,
            symbol_map=WATER_SYMBOL_MAP,
            category_orders={
                "base_system": EXPECTED_CROPPING_SYSTEMS,
                "water_regime": EXPECTED_WATER_REGIMES,
            },
            labels={
                "year": "Year",
                selected_annual_economic_col: f"{selected_economic_label} ($/acre)",
                "base_system": "Cropping system",
                "water_regime": "Water regime",
            },
            title=f"Annual {selected_economic_label.lower()}, 1981–2018",
        )
        overview_trend_fig.update_layout(height=500, legend_title_text="System / water regime")
        overview_trend_fig.update_traces(marker={"size": 7}, line={"width": 2.2})
        style_numeric_axis(overview_trend_fig, money=True)
        st.plotly_chart(overview_trend_fig, width="stretch")
    st.caption(selected_economic["definition"])

    st.subheader("Long-term summary")
    overview_cols = [
        "cropping_system",
        "years",
        "mean_yield_kg_ha",
        "yield_cv_pct",
        "mean_irrigation_mm",
        "incremental_iwue_kg_m3",
        selected_mean_economic_col,
        selected_total_economic_col,
    ]
    overview_cols = list(dict.fromkeys(overview_cols))
    st.dataframe(
        summary[overview_cols].sort_values(selected_mean_economic_col, ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "cropping_system": "Cropping system",
            "years": "Years",
            "mean_yield_kg_ha": st.column_config.NumberColumn("Mean yield (kg/ha)", format="%,.0f"),
            "yield_cv_pct": st.column_config.NumberColumn("Yield CV (%)", format="%.1f"),
            "mean_irrigation_mm": st.column_config.NumberColumn("Mean irrigation (mm/yr)", format="%.1f"),
            "incremental_iwue_kg_m3": st.column_config.NumberColumn("Incremental IWUE (kg/m³)", format="%.3f"),
            selected_mean_economic_col: st.column_config.NumberColumn(
                f"Average {selected_economic_label.lower()} ($/ac/yr)", format="$%,.0f"
            ),
            selected_total_economic_col: st.column_config.NumberColumn(
                f"Total {selected_economic_label.lower()} ($/ac)", format="$%,.0f"
            ),
        },
    )

with tabs[2]:
    st.subheader("Annual yield by crop and year")
    annual_yield_fig = crop_chart(
        crop_annual,
        y_col="yield_kg_ha",
        y_label="Annual yield (kg/ha)",
        title="Annual simulated yield by crop",
        all_systems_selected=all_systems_selected,
        markers=True,
    )
    if annual_yield_fig is None:
        st.info("Annual yield is unavailable for the current selection.")
    else:
        st.plotly_chart(annual_yield_fig, width="stretch")

    st.subheader(f"Annual {selected_economic_label.lower()} by crop and year")
    annual_econ_fig = crop_chart(
        crop_annual,
        y_col=selected_annual_economic_col,
        y_label=f"{selected_economic_label} ($/acre)",
        title=f"Annual {selected_economic_label.lower()} by crop",
        all_systems_selected=all_systems_selected,
        markers=True,
        money_axis=True,
    )
    if annual_econ_fig is None:
        st.info(
            "This crop-level economic measure is unavailable for the current selection. "
            "Marginal measures require a matching rainfed baseline."
        )
    else:
        st.plotly_chart(annual_econ_fig, width="stretch")
    st.caption(selected_economic["definition"])

with tabs[3]:
    st.subheader("Cumulative yield by crop and year")
    cumulative_yield_fig = crop_chart(
        crop_cumulative,
        y_col="cumulative_yield_kg_ha",
        y_label="Cumulative yield (kg/ha)",
        title="Cumulative simulated yield by crop, 1981–2018",
        all_systems_selected=all_systems_selected,
    )
    if cumulative_yield_fig is None:
        st.info("Cumulative yield is unavailable for the current selection.")
    else:
        st.plotly_chart(cumulative_yield_fig, width="stretch")

    st.subheader("Cumulative gross return by crop and year")
    cumulative_gross_fig = crop_chart(
        crop_cumulative,
        y_col="cumulative_gross_return_usd_ac",
        y_label="Cumulative gross return ($/acre)",
        title="Cumulative gross return by crop, 1981–2018",
        all_systems_selected=all_systems_selected,
        money_axis=True,
    )
    if cumulative_gross_fig is None:
        st.info("Cumulative gross return is unavailable for the current selection.")
    else:
        st.plotly_chart(cumulative_gross_fig, width="stretch")

    st.subheader("Cumulative return after operational cost by crop and year")
    cumulative_operating_fig = crop_chart(
        crop_cumulative,
        y_col="cumulative_return_after_operational_cost_usd_ac",
        y_label="Cumulative return after operational cost ($/acre)",
        title="Cumulative return after operational cost by crop, 1981–2018",
        all_systems_selected=all_systems_selected,
        money_axis=True,
    )
    if cumulative_operating_fig is None:
        st.info("Cumulative return after operational cost is unavailable for the current selection.")
    else:
        st.plotly_chart(cumulative_operating_fig, width="stretch")

    st.caption(
        "For rotations, a crop contributes only in the years when it is planted. "
        "Flat segments indicate years assigned to another crop. Cumulative yield is the sum of annual simulated kg/ha values for that crop; economic values are per acre."
    )

    cumulative_download_cols = [
        "year",
        "base_system",
        "water_regime",
        "crop_code",
        "crop_name",
        "yield_kg_ha",
        "cumulative_yield_kg_ha",
        "gross_income_simulated_usd_ac",
        "cumulative_gross_return_usd_ac",
        "return_above_operating_usd_ac",
        "cumulative_return_after_operational_cost_usd_ac",
        "crop_observed_this_year",
    ]
    st.download_button(
        "Download cumulative crop-year data",
        data=crop_cumulative[cumulative_download_cols].to_csv(index=False).encode("utf-8"),
        file_name="cumulative_crop_year_performance.csv",
        mime="text/csv",
    )

with tabs[4]:
    st.subheader(f"Annual {selected_economic_label.lower()}")
    annual_economic_data = annual.dropna(subset=[selected_annual_economic_col]).copy()
    if annual_economic_data.empty:
        st.info(
            "This economic measure is unavailable for the current selection. "
            "Marginal measures require a matching rainfed baseline."
        )
    else:
        annual_economic_fig = px.line(
            annual_economic_data,
            x="year",
            y=selected_annual_economic_col,
            color="base_system",
            line_dash="water_regime",
            symbol="water_regime",
            markers=True,
            color_discrete_map=SYSTEM_COLOR_MAP,
            line_dash_map=WATER_DASH_MAP,
            symbol_map=WATER_SYMBOL_MAP,
            category_orders={
                "base_system": EXPECTED_CROPPING_SYSTEMS,
                "water_regime": EXPECTED_WATER_REGIMES,
            },
            labels={
                "year": "Year",
                selected_annual_economic_col: f"{selected_economic_label} ($/acre)",
                "base_system": "Cropping system",
                "water_regime": "Water regime",
            },
            title=f"Annual {selected_economic_label.lower()}",
        )
        annual_economic_fig.update_layout(legend_title_text="System / water regime")
        annual_economic_fig.update_traces(marker={"size": 7}, line={"width": 2.2})
        style_numeric_axis(annual_economic_fig, money=True)
        st.plotly_chart(annual_economic_fig, width="stretch")

    st.subheader("Average and total economics")
    income_cols = [
        "cropping_system",
        "years",
        "mean_gross_income_usd_ac",
        "total_gross_income_usd_ac",
        "mean_return_above_operating_usd_ac",
        "total_return_above_operating_usd_ac",
        "mean_return_above_total_usd_ac",
        "total_return_above_total_usd_ac",
        "mean_marginal_gross_return_usd_ac",
        "mean_marginal_return_above_operating_usd_ac",
        "mean_marginal_return_above_total_usd_ac",
    ]
    st.dataframe(summary[income_cols], width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("Water use and irrigation water-use efficiency")
    st.caption(
        "This section compares rainfed and irrigated records for the cropping system selected in the sidebar, "
        "even when only one water regime is selected for the other dashboard tabs."
    )

    water_detail = detailed_all_regimes[
        detailed_all_regimes["water_regime"].isin(["Rainfed", "Irrigated"])
    ].copy()
    water_crop_annual = aggregate_crop_year(water_detail)
    irrigated_crop_years = water_crop_annual[
        water_crop_annual["water_regime"].eq("Irrigated")
        & (pd.to_numeric(water_crop_annual["irrigation_mm"], errors="coerce").fillna(0) > 0)
    ].copy()

    mean_irrigation = (
        irrigated_crop_years["irrigation_mm"].mean()
        if not irrigated_crop_years.empty
        else np.nan
    )
    total_irrigation = (
        pd.to_numeric(irrigated_crop_years["irrigation_mm"], errors="coerce").fillna(0).sum()
        if not irrigated_crop_years.empty
        else 0.0
    )
    mean_productivity = (
        irrigated_crop_years["irrigation_productivity_kg_m3"].mean()
        if "irrigation_productivity_kg_m3" in irrigated_crop_years.columns
        else np.nan
    )
    mean_iwue = (
        irrigated_crop_years["incremental_iwue_kg_m3"].mean()
        if "incremental_iwue_kg_m3" in irrigated_crop_years.columns
        else np.nan
    )

    st.subheader("Rainfed and irrigated yield by crop and year")
    water_yield_fig = crop_chart(
        water_crop_annual,
        y_col="yield_kg_ha",
        y_label="Yield (kg/ha)",
        title="Annual yield under rainfed and irrigated production",
        all_systems_selected=all_systems_selected,
        markers=True,
    )
    if water_yield_fig is None:
        st.info("Rainfed–irrigated yield comparison is unavailable for this system.")
    else:
        st.plotly_chart(water_yield_fig, width="stretch")

    if irrigated_crop_years.empty:
        st.info(
            "No positive applied-irrigation records are available for the selected cropping system. "
            "Irrigation productivity and incremental IWUE cannot be calculated."
        )
    else:
        st.subheader("Applied irrigation by crop and year")
        irrigation_fig = crop_chart(
            irrigated_crop_years,
            y_col="irrigation_mm",
            y_label="Applied irrigation (mm/crop-year)",
            title="Annual applied irrigation",
            all_systems_selected=all_systems_selected,
            markers=True,
        )
        if irrigation_fig is not None:
            st.plotly_chart(irrigation_fig, width="stretch")

        water_line_c1, water_line_c2 = st.columns(2)
        with water_line_c1:
            st.subheader("Irrigation productivity")
            productivity_fig = crop_chart(
                irrigated_crop_years,
                y_col="irrigation_productivity_kg_m3",
                y_label="Irrigation productivity (kg/m³)",
                title="Yield produced per unit of applied irrigation",
                all_systems_selected=all_systems_selected,
                markers=True,
            )
            if productivity_fig is None:
                st.info("Irrigation productivity is unavailable.")
            else:
                st.plotly_chart(productivity_fig, width="stretch")

        with water_line_c2:
            st.subheader("Incremental IWUE")
            iwue_fig = crop_chart(
                irrigated_crop_years,
                y_col="incremental_iwue_kg_m3",
                y_label="Incremental IWUE (kg/m³)",
                title="Yield gain above rainfed per unit of irrigation",
                all_systems_selected=all_systems_selected,
                markers=True,
            )
            if iwue_fig is None:
                st.info(
                    "Incremental IWUE requires a matching rainfed crop-system-year baseline."
                )
            else:
                st.plotly_chart(iwue_fig, width="stretch")

    st.subheader("Water-use summary indicators")
    water_c1, water_c2, water_c3, water_c4 = st.columns(4)
    water_c1.metric(
        "Mean irrigation",
        "Not available" if pd.isna(mean_irrigation) else f"{mean_irrigation:,.1f} mm/crop-year",
    )
    water_c2.metric("Cumulative irrigation", f"{total_irrigation:,.0f} mm")
    water_c3.metric(
        "Irrigation productivity",
        "Not available" if pd.isna(mean_productivity) else f"{mean_productivity:,.3f} kg/m³",
        help="Irrigated yield divided by applied irrigation volume.",
    )
    water_c4.metric(
        "Incremental IWUE",
        "Not available" if pd.isna(mean_iwue) else f"{mean_iwue:,.3f} kg/m³",
        help="Site/system yield gain above the paired rainfed yield divided by applied irrigation volume.",
    )

    st.subheader("Water use and irrigation water-use efficiency table")
    water_by_crop = (
        water_crop_annual.groupby(
            ["cropping_system", "base_system", "water_regime", "crop_code", "crop_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            crop_years=("year", "nunique"),
            mean_yield_kg_ha=("yield_kg_ha", "mean"),
            mean_irrigation_mm=("irrigation_mm", "mean"),
            total_irrigation_mm=("irrigation_mm", "sum"),
            irrigation_productivity_kg_m3=("irrigation_productivity_kg_m3", "mean"),
            incremental_iwue_kg_m3=("incremental_iwue_kg_m3", "mean"),
        )
    )
    st.dataframe(
        water_by_crop,
        width="stretch",
        hide_index=True,
        column_config={
            "cropping_system": "Cropping system / regime",
            "base_system": "Cropping system",
            "water_regime": "Water regime",
            "crop_code": "Crop code",
            "crop_name": "Crop",
            "crop_years": "Crop-years",
            "mean_yield_kg_ha": st.column_config.NumberColumn("Mean yield (kg/ha)", format="%,.0f"),
            "mean_irrigation_mm": st.column_config.NumberColumn("Mean irrigation (mm/crop-year)", format="%.1f"),
            "total_irrigation_mm": st.column_config.NumberColumn("Cumulative irrigation (mm)", format="%,.0f"),
            "irrigation_productivity_kg_m3": st.column_config.NumberColumn("Irrigation productivity (kg/m³)", format="%.3f"),
            "incremental_iwue_kg_m3": st.column_config.NumberColumn("Incremental IWUE (kg/m³)", format="%.3f"),
        },
    )


    st.caption(
        "Irrigation productivity = irrigated yield ÷ irrigation volume. "
        "Incremental IWUE = (irrigated yield − paired rainfed yield) ÷ irrigation volume. "
        "Because 1 mm over 1 hectare equals 10 m³, both indicators are reported in kg/m³."
    )

with tabs[6]:
    st.subheader("Producer optimizer: compare all cropping systems")
    st.caption(
        "The chart compares every available cropping system and water regime in the processed dataset. "
        "The single-system selector used in the other tabs does not limit this comparison."
    )

    preferred_objective = "Highest average return above operating cost"
    objective_options = [
        label
        for label, (metric, _ascending) in OBJECTIVES.items()
        if metric in optimizer_summary_all.columns
        and optimizer_summary_all[metric].notna().any()
    ]
    if st.session_state.get("optimizer_objective") not in objective_options:
        st.session_state["optimizer_objective"] = (
            preferred_objective
            if preferred_objective in objective_options
            else objective_options[0]
        )
    objective = st.selectbox(
        "Optimization objective",
        objective_options,
        key="optimizer_objective",
        help=(
            "Average return after operating cost is the recommended general comparison because it is "
            "available for all crop systems. Water-efficiency objectives require irrigated and paired rainfed data."
        ),
    )

    with st.expander("Eligibility and optional producer constraints", expanded=False):
        eligibility_c1, eligibility_c2 = st.columns([2, 1])
        with eligibility_c1:
            optimizer_regime = st.selectbox(
                "Eligible water regimes",
                ["All water regimes", *available_regimes],
                key="optimizer_regime",
            )
        with eligibility_c2:
            complete_only = st.checkbox(
                "Require all 38 years",
                value=False,
                key="optimizer_complete_only",
                help="Use only choices represented from 1981 through 2018.",
            )

        constraint_col1, constraint_col2 = st.columns(2)
        with constraint_col1:
            use_water_limit = st.checkbox(
                "Set maximum average irrigation",
                key="optimizer_use_water_limit",
            )
            water_limit = st.number_input(
                "Maximum average irrigation (mm/year)",
                min_value=0.0,
                value=300.0,
                step=10.0,
                disabled=not use_water_limit,
                key="optimizer_water_limit",
            )
        with constraint_col2:
            use_return_floor = st.checkbox(
                "Set minimum operational return",
                key="optimizer_use_return_floor",
            )
            return_floor = st.number_input(
                "Minimum average return after operational cost ($/acre/year)",
                value=0.0,
                step=10.0,
                disabled=not use_return_floor,
                key="optimizer_return_floor",
            )

    optimizer_scope = optimizer_summary_all.copy()
    if optimizer_regime != "All water regimes":
        optimizer_scope = optimizer_scope[
            optimizer_scope["water_regime"].eq(optimizer_regime)
        ]
    if complete_only:
        optimizer_scope = optimizer_scope[optimizer_scope["years"].eq(38)]

    metric_col, ascending = OBJECTIVES[objective]
    ranking = rank_systems(
        optimizer_scope,
        objective,
        max_mean_irrigation_mm=water_limit if use_water_limit else None,
        min_mean_return_operating=return_floor if use_return_floor else None,
    )

    # Each optimizer objective has an explicit axis title and a non-compact
    # tick format. This prevents Plotly abbreviations such as 1k and 500m and
    # makes the physical/economic unit unambiguous for every comparison.
    objective_units = {
        "mean_yield_kg_ha": {
            "value_kind": "numeric",
            "suffix": " kg/ha",
            "decimals": 0,
            "axis_title": "Mean yield (kg/ha)",
            "tickformat": ",.0f",
        },
        "incremental_iwue_kg_m3": {
            "value_kind": "numeric",
            "suffix": " kg/m³",
            "decimals": 3,
            "axis_title": "Irrigation water-use efficiency (kg/m³)",
            "tickformat": ",.3f",
        },
        "irrigation_productivity_kg_m3": {
            "value_kind": "numeric",
            "suffix": " kg/m³",
            "decimals": 3,
            "axis_title": "Irrigation productivity (kg/m³)",
            "tickformat": ",.3f",
        },
        "mean_gross_income_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre/year",
            "decimals": 0,
            "axis_title": "Average gross income ($/acre/year)",
            "tickformat": ",.0f",
        },
        "total_gross_income_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre",
            "decimals": 0,
            "axis_title": "Total gross income ($/acre)",
            "tickformat": ",.0f",
        },
        "mean_return_above_operating_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre/year",
            "decimals": 0,
            "axis_title": "Average return above operating cost ($/acre/year)",
            "tickformat": ",.0f",
        },
        "total_return_above_operating_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre",
            "decimals": 0,
            "axis_title": "Total return above operating cost ($/acre)",
            "tickformat": ",.0f",
        },
        "mean_return_above_total_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre/year",
            "decimals": 0,
            "axis_title": "Average return above total cost ($/acre/year)",
            "tickformat": ",.0f",
        },
        "mean_marginal_return_above_operating_usd_ac": {
            "value_kind": "money",
            "suffix": "/acre/year",
            "decimals": 0,
            "axis_title": "Average marginal return above operating cost ($/acre/year)",
            "tickformat": ",.0f",
        },
        "mean_marginal_return_per_mm_usd_ac_mm": {
            "value_kind": "money",
            "suffix": "/acre/mm",
            "decimals": 3,
            "axis_title": "Marginal return per mm irrigation ($/acre/mm)",
            "tickformat": ",.3f",
        },
        "mean_irrigation_mm": {
            "value_kind": "numeric",
            "suffix": " mm/year",
            "decimals": 1,
            "axis_title": "Average irrigation (mm/year)",
            "tickformat": ",.1f",
        },
        "yield_cv_pct": {
            "value_kind": "numeric",
            "suffix": "%",
            "decimals": 1,
            "axis_title": "Yield variability (%)",
            "tickformat": ",.1f",
        },
    }
    unit_spec = objective_units.get(
        metric_col,
        {
            "value_kind": "numeric",
            "suffix": "",
            "decimals": 2,
            "axis_title": objective,
            "tickformat": ",.2f",
        },
    )
    value_kind = unit_spec["value_kind"]
    suffix = unit_spec["suffix"]
    decimals = unit_spec["decimals"]
    axis_title = unit_spec["axis_title"]
    axis_tickformat = unit_spec["tickformat"]

    if ranking.empty:
        st.warning(
            "No cropping-system choice satisfies the selected eligibility rules and constraints. "
            "Open the eligibility section and remove the 38-year requirement or optional limits."
        )
        st.button("Reset optimizer", on_click=reset_optimizer_state)
    else:
        ranking = ranking.copy()
        ranking["system_display"] = (
            ranking["base_system"] + " | " + ranking["water_regime"]
        )

        chart_data = ranking.dropna(subset=[metric_col]).copy()
        present_systems = set(chart_data["base_system"].astype(str))
        system_order = [
            system for system in EXPECTED_CROPPING_SYSTEMS if system in present_systems
        ]
        chart_data["base_system"] = pd.Categorical(
            chart_data["base_system"], categories=system_order, ordered=True
        )
        chart_data = chart_data.sort_values(
            ["base_system", "water_regime"], ascending=[False, True]
        )

        rank_fig = px.bar(
            chart_data,
            x=metric_col,
            y="base_system",
            orientation="h",
            color="water_regime",
            barmode="group",
            text=metric_col,
            category_orders={
                "base_system": system_order,
                "water_regime": EXPECTED_WATER_REGIMES,
            },
            labels={
                metric_col: axis_title,
                "base_system": "Cropping system",
                "water_regime": "Water regime",
            },
            title=f"All cropping systems: {objective}",
        )
        if value_kind == "money":
            rank_fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
        elif decimals == 0:
            rank_fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside")
        else:
            rank_fig.update_traces(
                texttemplate=f"%{{x:,.{decimals}f}}", textposition="outside"
            )
        rank_fig.update_layout(
            height=max(500, 62 * len(system_order)),
            legend_title_text="Water regime",
            bargap=0.20,
            bargroupgap=0.06,
            margin={"l": 70, "r": 85, "t": 65, "b": 45},
        )
        rank_fig.update_xaxes(
            title_text=axis_title,
            tickformat=axis_tickformat,
            tickprefix="$" if value_kind == "money" else "",
            separatethousands=True,
            exponentformat="none",
            showexponent="none",
        )
        st.plotly_chart(rank_fig, width="stretch")

        systems_in_scope = set(optimizer_scope["base_system"].astype(str))
        missing_from_chart = [
            system
            for system in EXPECTED_CROPPING_SYSTEMS
            if system in systems_in_scope and system not in present_systems
        ]
        if missing_from_chart:
            st.info(
                "No valid value exists for this objective for: "
                + ", ".join(missing_from_chart)
                + ". Choose average return after operating cost to compare all systems."
            )

        best = ranking.iloc[0]
        best_value = best[metric_col]
        best_value_text = (
            compact_money(best_value, suffix)
            if value_kind == "money"
            else f"{best_value:,.{decimals}f}{suffix}"
        )
        best_c1, best_c2, best_c3, best_c4 = st.columns(4)
        best_c1.metric("Recommended choice", best["system_display"])
        best_c2.metric("Objective value", best_value_text)
        best_c3.metric("Mean yield", f"{best['mean_yield_kg_ha']:,.0f} kg/ha")
        best_c4.metric(
            "Return after operational cost",
            compact_money(best["mean_return_above_operating_usd_ac"], "/acre/year"),
        )

        if "total" in objective.lower() and not complete_only and ranking["years"].nunique() > 1:
            st.warning(
                "Total-return rankings include choices with different numbers of available years. "
                "Select ‘Require all 38 years’ for a full-period comparison."
            )

        st.subheader("Producer comparison table")
        display_cols = list(
            dict.fromkeys(
                [
                    "rank",
                    "base_system",
                    "water_regime",
                    "years",
                    metric_col,
                    "mean_yield_kg_ha",
                    "yield_cv_pct",
                    "mean_irrigation_mm",
                    "incremental_iwue_kg_m3",
                    "mean_return_above_operating_usd_ac",
                    "mean_return_above_total_usd_ac",
                ]
            )
        )
        st.dataframe(
            ranking[display_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "rank": "Rank",
                "base_system": "Cropping system",
                "water_regime": "Water regime",
                "years": "Years",
                "mean_yield_kg_ha": st.column_config.NumberColumn("Mean yield (kg/ha)", format="%,.0f"),
                "yield_cv_pct": st.column_config.NumberColumn("Yield CV (%)", format="%.1f"),
                "mean_irrigation_mm": st.column_config.NumberColumn("Mean irrigation (mm/yr)", format="%.1f"),
                "incremental_iwue_kg_m3": st.column_config.NumberColumn("Incremental IWUE (kg/m³)", format="%.3f"),
                "mean_return_above_operating_usd_ac": st.column_config.NumberColumn("Average return after operational cost ($/ac/yr)", format="$%,.0f"),
                "mean_return_above_total_usd_ac": st.column_config.NumberColumn("Average return after total cost ($/ac/yr)", format="$%,.0f"),
            },
        )
        st.download_button(
            "Download producer comparison",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="producer_optimizer_all_systems.csv",
            mime="text/csv",
        )

with tabs[7]:
    st.subheader("Simulation-data coverage")
    coverage = (
        detailed.groupby(["cropping_system", "crop_code"], as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            years_with_crop=("year", "nunique"),
            n_sites=("n_sites", "max"),
            missing_yield=("yield_kg_ha", lambda values: int(values.isna().sum())),
            missing_irrigation=("irrigation_mm", lambda values: int(values.isna().sum())),
        )
    )
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.subheader("Economic-data coverage")
    economic_coverage = (
        economics.groupby(["crop_code", "crop_name"], as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            years=("year", "nunique"),
            missing_price=("price_usd_bu", lambda values: int(values.isna().sum())),
            missing_operating_cost=("operating_cost_usd_ac", lambda values: int(values.isna().sum())),
            missing_total_cost=("total_production_cost_usd_ac", lambda values: int(values.isna().sum())),
        )
    )
    st.dataframe(economic_coverage, width="stretch", hide_index=True)

    st.subheader("Methods and interpretation")
    st.markdown(
        """
- **One-system display:** the Cropping system selector shows one system at a time by default. “All cropping systems” enables comparisons and ranking.
- **Yield:** the DSSAT/statewide simulated yield supplied in the processed files is used directly in kg/ha.
- **Bushel conversion:** 56 lb/bu is used for maize and sorghum; 60 lb/bu is used for soybean and wheat.
- **Gross return:** simulated yield converted to bu/acre × the supplied annual crop price.
- **Return after operational cost:** simulated gross return − the supplied annual operational cost.
- **Return after total production cost:** simulated gross return − the supplied annual total production cost.
- **Surveyed economics:** the source workbook also contains surveyed yield and surveyed returns. Those values describe the survey crop, while the dashboard recalculates returns from the simulated yield so cropping systems can be compared consistently.
- **Cumulative crop yield:** annual yield is summed separately for each crop. In rotations, flat years indicate that another crop occupied that year.
- **Dollar basis:** annual values are treated as nominal dollars and all economic results are per acre.
"""
    )


with tabs[1]:
    st.subheader("Spatial farm maps")
    if not spatial_available:
        st.info(
            "The spatial site-year file has not been created. Rerun "
            "prepare_H_drive_data.ps1 to create cropping_systems_spatial.parquet."
        )
    elif not spatial_enabled:
        st.info(
            "The selected cropping system does not have an available rainfed or irrigated year. "
            "Choose another system in Spatial farm options."
        )
    else:
        with st.spinner("Preparing the fully filled Kansas map..."):
            spatial_simulation = cached_load_spatial(str(SPATIAL_PATH))
            master_sites = build_complete_master_grid(
                spatial_simulation[["site_id", "latitude", "longitude"]],
                max_grid_sites=5000,
            )

            spatial_metric_col, spatial_metric_label, spatial_money = SPATIAL_MEASURES[
                spatial_measure
            ]
            baseline_metrics = {
                "marginal_gross_return_usd_ac",
                "marginal_return_above_operating_usd_ac",
                "incremental_iwue_kg_m3",
            }
            source_regimes = {spatial_regime}
            if spatial_regime == "Irrigated" and spatial_metric_col in baseline_metrics:
                source_regimes.add("Rainfed")

            spatial_metric_source = spatial_simulation[
                spatial_simulation["base_system"].eq(spatial_system)
                & spatial_simulation["year"].eq(spatial_year)
                & spatial_simulation["water_regime"].isin(source_regimes)
            ].copy()

            if spatial_metric_source.empty:
                spatial_map_data = pd.DataFrame()
                direct_site_count = 0
            else:
                spatial_detailed = add_economic_and_water_metrics(
                    spatial_metric_source, economics
                )
                spatial_selected = spatial_detailed[
                    spatial_detailed["water_regime"].eq(spatial_regime)
                ].copy()

                if spatial_metric_col not in spatial_selected.columns:
                    spatial_map_data = pd.DataFrame()
                    direct_site_count = 0
                else:
                    spatial_observed = (
                        spatial_selected.groupby("site_id", as_index=False)
                        .agg(
                            latitude=("latitude", "median"),
                            longitude=("longitude", "median"),
                            map_value=(spatial_metric_col, "mean"),
                        )
                        .rename(columns={"map_value": spatial_metric_col})
                    )
                    finite_direct = pd.to_numeric(
                        spatial_observed[spatial_metric_col], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan)
                    direct_site_count = int(finite_direct.notna().sum())
                    spatial_filled_map = idw_fill_complete_grid(
                        master_sites,
                        spatial_observed,
                        spatial_metric_col,
                        neighbors=16,
                    )
                    spatial_map_data = spatial_filled_map

        if spatial_map_data.empty:
            if spatial_metric_col in baseline_metrics and spatial_regime == "Irrigated":
                st.warning(
                    "This measure requires a same-site rainfed baseline. No matching rainfed baseline "
                    "is available for the selected cropping system and year. Choose yield, gross return, "
                    "net return, applied irrigation, or IWUE instead."
                )
            else:
                st.warning(
                    "No numeric values are available for this system, year, water regime, and measure. "
                    "Choose another combination."
                )
        else:
            values = pd.to_numeric(
                spatial_map_data[spatial_metric_col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            value_min = float(values.min())
            value_max = float(values.max())
            observed_cells = int(spatial_map_data["fill_method"].eq("observed").sum())
            filled_cells = int(spatial_map_data["fill_method"].eq("idw").sum())

            style = SPATIAL_COLOR_STYLES[spatial_style]
            crosses_zero = value_min < 0 < value_max
            # Color scaling is recalculated from the actual values in the selected year,
            # system, regime, and measure so each year has its own within-map contrast.
            if crosses_zero:
                limit = max(abs(value_min), abs(value_max))
                cmin, cmax = -limit, limit
                color_scale = [
                    [0.00, "#c81d25"],
                    [0.22, "#e67e5f"],
                    [0.50, "#f2efc8"],
                    [0.78, "#9ecf74"],
                    [1.00, "#1a9850"],
                ]
                cmid = 0
            else:
                cmin, cmax = value_min, value_max
                color_scale = style["scale"]
                cmid = None
                if np.isclose(cmin, cmax):
                    cmin = value_min - 1e-9
                    cmax = value_max + 1e-9

            plot_data = spatial_map_data.sort_values(["latitude", "longitude"]).reset_index(drop=True)
            customdata = np.column_stack(
                [
                    plot_data["site_id"].astype(str).to_numpy(),
                    plot_data["fill_method"].astype(str).to_numpy(),
                ]
            )
            hover_value = (
                "$%{marker.color:,.2f}/acre"
                if spatial_money
                else "%{marker.color:,.2f}"
            )
            spatial_fig = go.Figure(
                go.Scattergl(
                    x=plot_data["longitude"],
                    y=plot_data["latitude"],
                    mode="markers",
                    marker={
                        "symbol": "square",
                        "size": 15.5,
                        "opacity": 1.0,
                        "color": plot_data[spatial_metric_col],
                        "colorscale": color_scale,
                        "cmin": cmin,
                        "cmax": cmax,
                        "cmid": cmid,
                        "showscale": True,
                        "line": {"width": 0.35, "color": style["line_color"]},
                        "colorbar": {
                            "title": spatial_metric_label,
                            "tickprefix": "$" if spatial_money else "",
                            "tickformat": "~s",
                        },
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "Site: %{customdata[0]}<br>"
                        "Longitude: %{x:.4f}<br>"
                        "Latitude: %{y:.4f}<br>"
                        + spatial_metric_label
                        + ": "
                        + hover_value
                        + "<extra></extra>"
                    ),
                )
            )
            spatial_fig.update_layout(
                title=(
                    f"{spatial_system} | {spatial_regime} | {spatial_year}: "
                    f"{spatial_measure}"
                ),
                height=760,
                margin={"l": 35, "r": 25, "t": 55, "b": 35},
                paper_bgcolor=style["background"],
                plot_bgcolor=style["background"],
                xaxis={
                    "title": "Longitude",
                    "showgrid": False,
                    "zeroline": False,
                    "range": [
                        float(plot_data["longitude"].min()) - 0.08,
                        float(plot_data["longitude"].max()) + 0.08,
                    ],
                },
                yaxis={
                    "title": "Latitude",
                    "showgrid": False,
                    "zeroline": False,
                    "scaleanchor": "x",
                    "scaleratio": 1.25,
                    "range": [
                        float(plot_data["latitude"].min()) - 0.08,
                        float(plot_data["latitude"].max()) + 0.08,
                    ],
                },
            )
            st.plotly_chart(spatial_fig, width="stretch")

            st.caption(
                "The map uses the actual 2,776 simulated observation sites as the data source. "
                "Observed site values come directly from the actual 2,776 simulation sites. Only unobserved internal display cells are interpolated. The color scale is recalculated separately for the selected year, system, regime, and measure, with red for low values and green for high values."
            )
            with st.expander("Map details", expanded=False):
                st.write(
                    f"Actual observed simulation sites used: {observed_cells:,}. "
                    f"Completed grid cells shown after filling internal gaps: {len(plot_data):,}. "
                    f"Interpolated internal cells only: {filled_cells:,}."
                )
                if np.isclose(value_min, value_max):
                    st.write(
                        "This selected field is nearly spatially uniform for the chosen year. "
                        "Choose yield, gross return, or return after cost to see stronger spatial contrast."
                    )
