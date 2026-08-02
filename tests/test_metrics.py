import pandas as pd

from src.metrics import (
    add_economic_and_water_metrics,
    aggregate_system_year,
    kg_ha_to_bu_ac,
    rank_systems,
    summarize_systems,
)


def test_kg_ha_to_bu_ac_maize():
    result = kg_ha_to_bu_ac(
        pd.Series([10000.0]),
        pd.Series(["MZ"]),
    )
    assert 159.0 < result.iloc[0] < 160.0


def test_incremental_iwue_and_returns():
    simulation = pd.DataFrame(
        [
            {
                "year": 2000,
                "cropping_system": "MZ | Rainfed",
                "base_system": "MZ",
                "rotation": "MZ",
                "crop_code": "MZ",
                "water_regime": "Rainfed",
                "yield_kg_ha": 5000.0,
                "irrigation_mm": 0.0,
                "rainfall_mm": 500.0,
                "n_sites": 10,
            },
            {
                "year": 2000,
                "cropping_system": "MZ | Irrigated",
                "base_system": "MZ",
                "rotation": "MZ",
                "crop_code": "MZ",
                "water_regime": "Irrigated",
                "yield_kg_ha": 9000.0,
                "irrigation_mm": 200.0,
                "rainfall_mm": 500.0,
                "n_sites": 10,
            },
        ]
    )
    economics = pd.DataFrame(
        [
            {
                "year": 2000,
                "crop_code": "MZ",
                "price_usd_bu": 2.0,
                "operating_cost_usd_ac": 100.0,
                "total_production_cost_usd_ac": 200.0,
                "gross_value_survey_usd_ac": 0.0,
                "return_above_operating_survey_usd_ac": 0.0,
                "return_above_total_survey_usd_ac": 0.0,
                "survey_yield_bu_ac": 0.0,
                "crop_name": "Maize",
            }
        ]
    )
    result = add_economic_and_water_metrics(simulation, economics)
    irrigated = result[result["water_regime"] == "Irrigated"].iloc[0]
    assert abs(irrigated["incremental_iwue_kg_m3"] - 2.0) < 1e-9
    assert irrigated["gross_income_simulated_usd_ac"] > 0
    assert irrigated["marginal_gross_return_usd_ac"] > 0
    assert abs(
        irrigated["marginal_return_above_operating_usd_ac"]
        - irrigated["marginal_gross_return_usd_ac"]
    ) < 1e-9
    assert irrigated["marginal_return_per_mm_usd_ac_mm"] > 0
    assert irrigated["crop_name"] == "Maize"
    assert "crop_name_x" not in result.columns
    assert "crop_name_y" not in result.columns


def test_summary_and_ranking():
    annual = pd.DataFrame(
        [
            {
                "cropping_system": "A",
                "base_system": "A",
                "rotation": "A",
                "water_regime": "Rainfed",
                "year": 2000,
                "yield_kg_ha": 1000.0,
                "yield_bu_ac": 10.0,
                "irrigation_mm": 0.0,
                "rainfall_mm": 500.0,
                "gross_income_simulated_usd_ac": 100.0,
                "return_above_operating_usd_ac": 40.0,
                "return_above_total_usd_ac": 10.0,
                "marginal_gross_return_usd_ac": 0.0,
                "marginal_return_above_operating_usd_ac": 0.0,
                "marginal_return_above_total_usd_ac": 0.0,
                "marginal_return_per_mm_usd_ac_mm": float("nan"),
                "irrigation_productivity_kg_m3": float("nan"),
                "incremental_iwue_kg_m3": float("nan"),
                "n_sites": 5,
            },
            {
                "cropping_system": "B",
                "base_system": "B",
                "rotation": "B",
                "water_regime": "Irrigated",
                "year": 2000,
                "yield_kg_ha": 2000.0,
                "yield_bu_ac": 20.0,
                "irrigation_mm": 100.0,
                "rainfall_mm": 500.0,
                "gross_income_simulated_usd_ac": 200.0,
                "return_above_operating_usd_ac": 80.0,
                "return_above_total_usd_ac": 20.0,
                "marginal_gross_return_usd_ac": 100.0,
                "marginal_return_above_operating_usd_ac": 40.0,
                "marginal_return_above_total_usd_ac": 10.0,
                "marginal_return_per_mm_usd_ac_mm": 0.4,
                "irrigation_productivity_kg_m3": 2.0,
                "incremental_iwue_kg_m3": 1.0,
                "n_sites": 5,
            },
        ]
    )
    summary = summarize_systems(annual)
    ranked = rank_systems(summary, "Highest mean yield")
    assert ranked.iloc[0]["cropping_system"] == "B"
    assert ranked.iloc[0]["rank"] == 1


def test_crop_cumulative_metrics_keep_flat_rotation_years():
    from src.metrics import add_crop_cumulative_metrics

    crop_annual = pd.DataFrame(
        {
            "cropping_system": ["SB-MZ | Rainfed", "SB-MZ | Rainfed"],
            "base_system": ["SB-MZ", "SB-MZ"],
            "rotation": ["SB-MZ", "SB-MZ"],
            "water_regime": ["Rainfed", "Rainfed"],
            "crop_code": ["SB", "SB"],
            "crop_name": ["Soybean", "Soybean"],
            "year": [1981, 1983],
            "yield_kg_ha": [1000.0, 1200.0],
            "gross_income_simulated_usd_ac": [200.0, 250.0],
            "return_above_operating_usd_ac": [80.0, 100.0],
        }
    )
    result = add_crop_cumulative_metrics(crop_annual, 1981, 1983)
    assert result["year"].tolist() == [1981, 1982, 1983]
    assert result["cumulative_yield_kg_ha"].tolist() == [1000.0, 1000.0, 2200.0]
    assert result["cumulative_gross_return_usd_ac"].tolist() == [200.0, 200.0, 450.0]
    assert result["cumulative_return_after_operational_cost_usd_ac"].tolist() == [80.0, 80.0, 180.0]


def test_aggregate_crop_year_retains_marginal_economic_columns():
    from src.metrics import aggregate_crop_year

    detailed = pd.DataFrame(
        {
            "cropping_system": ["MZ | Irrigated"],
            "base_system": ["MZ"],
            "rotation": ["MZ"],
            "water_regime": ["Irrigated"],
            "crop_code": ["MZ"],
            "crop_name": ["Maize"],
            "year": [2000],
            "yield_kg_ha": [9000.0],
            "yield_bu_ac": [143.0],
            "irrigation_mm": [200.0],
            "gross_income_simulated_usd_ac": [300.0],
            "return_above_operating_usd_ac": [180.0],
            "return_above_total_usd_ac": [90.0],
            "marginal_gross_return_usd_ac": [100.0],
            "marginal_return_above_operating_usd_ac": [80.0],
            "marginal_return_above_total_usd_ac": [40.0],
            "marginal_return_per_mm_usd_ac_mm": [0.4],
        }
    )
    result = aggregate_crop_year(detailed)
    assert result.loc[0, "marginal_gross_return_usd_ac"] == 100.0
    assert result.loc[0, "marginal_return_above_operating_usd_ac"] == 80.0
    assert result.loc[0, "marginal_return_above_total_usd_ac"] == 40.0


def test_aggregate_crop_year_retains_water_efficiency_columns():
    from src.metrics import aggregate_crop_year

    df = pd.DataFrame(
        {
            "cropping_system": ["MZ | Irrigated"],
            "base_system": ["MZ"],
            "rotation": ["MZ"],
            "water_regime": ["Irrigated"],
            "crop_code": ["MZ"],
            "crop_name": ["Maize"],
            "year": [1981],
            "yield_kg_ha": [8000.0],
            "yield_bu_ac": [127.0],
            "irrigation_mm": [125.0],
            "gross_income_simulated_usd_ac": [500.0],
            "return_above_operating_usd_ac": [250.0],
            "return_above_total_usd_ac": [100.0],
            "marginal_gross_return_usd_ac": [120.0],
            "marginal_return_above_operating_usd_ac": [90.0],
            "marginal_return_above_total_usd_ac": [50.0],
            "marginal_return_per_mm_usd_ac_mm": [0.72],
            "irrigation_productivity_kg_m3": [6.4],
            "incremental_iwue_kg_m3": [1.2],
        }
    )
    out = aggregate_crop_year(df)
    assert out.loc[0, "irrigation_productivity_kg_m3"] == 6.4
    assert out.loc[0, "incremental_iwue_kg_m3"] == 1.2


def test_water_limit_excludes_unknown_irrigation():
    from src.metrics import rank_systems

    summary = pd.DataFrame(
        {
            "cropping_system": ["MZ | Rainfed", "WT | Potential"],
            "mean_irrigation_mm": [0.0, float("nan")],
            "mean_yield_kg_ha": [4000.0, 9000.0],
            "mean_return_above_operating_usd_ac": [100.0, 300.0],
        }
    )
    ranked = rank_systems(
        summary,
        "Highest mean yield",
        max_mean_irrigation_mm=100.0,
    )
    assert ranked["cropping_system"].tolist() == ["MZ | Rainfed"]
