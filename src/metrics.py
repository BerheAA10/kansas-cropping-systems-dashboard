from __future__ import annotations

import numpy as np
import pandas as pd

BUSHEL_WEIGHT_LB = {
    "MZ": 56.0,
    "SG": 56.0,
    "SB": 60.0,
    "WT": 60.0,
}

KG_HA_TO_LB_AC = 0.8921791216


def kg_ha_to_bu_ac(yield_kg_ha: pd.Series, crop_code: pd.Series) -> pd.Series:
    weights = crop_code.map(BUSHEL_WEIGHT_LB)
    return yield_kg_ha * KG_HA_TO_LB_AC / weights


def add_economic_and_water_metrics(
    simulation: pd.DataFrame,
    economics: pd.DataFrame,
    irrigation_cost_usd_ac_per_mm: float = 0.0,
) -> pd.DataFrame:
    """Join crop-year economics and calculate income and water-use indicators."""
    # ``crop_name`` is descriptive metadata present in both source tables.
    # Exclude it from the economics side so pandas does not create suffixes.
    economics_for_merge = economics.drop(columns=["crop_name"], errors="ignore")
    df = simulation.merge(
        economics_for_merge,
        on=["year", "crop_code"],
        how="left",
        validate="many_to_one",
    )

    crop_names = {
        "MZ": "Maize",
        "WT": "Wheat",
        "SG": "Sorghum",
        "SB": "Soybean",
    }
    if "crop_name" not in df.columns:
        df["crop_name"] = df["crop_code"].map(crop_names).fillna(df["crop_code"])
    else:
        df["crop_name"] = df["crop_name"].fillna(
            df["crop_code"].map(crop_names)
        )

    df["yield_bu_ac"] = kg_ha_to_bu_ac(df["yield_kg_ha"], df["crop_code"])
    df["gross_income_simulated_usd_ac"] = df["yield_bu_ac"] * df["price_usd_bu"]

    irrig_mm = df["irrigation_mm"].fillna(0).clip(lower=0)
    df["assumed_irrigation_cost_usd_ac"] = (
        irrig_mm * float(irrigation_cost_usd_ac_per_mm)
    )
    df["adjusted_operating_cost_usd_ac"] = (
        df["operating_cost_usd_ac"] + df["assumed_irrigation_cost_usd_ac"]
    )
    df["adjusted_total_cost_usd_ac"] = (
        df["total_production_cost_usd_ac"] + df["assumed_irrigation_cost_usd_ac"]
    )
    df["return_above_operating_usd_ac"] = (
        df["gross_income_simulated_usd_ac"] - df["adjusted_operating_cost_usd_ac"]
    )
    df["return_above_total_usd_ac"] = (
        df["gross_income_simulated_usd_ac"] - df["adjusted_total_cost_usd_ac"]
    )

    # Simple irrigation productivity. 1 mm over 1 ha = 10 m3.
    df["irrigation_productivity_kg_m3"] = np.where(
        irrig_mm > 0,
        df["yield_kg_ha"] / (irrig_mm * 10.0),
        np.nan,
    )

    # Pair irrigated/potential records with the matching rainfed crop-system-year.
    # These paired values support both incremental IWUE and marginal economics.
    baseline_metrics = {
        "yield_kg_ha": "paired_rainfed_yield_kg_ha",
        "gross_income_simulated_usd_ac": "paired_rainfed_gross_income_usd_ac",
        "return_above_operating_usd_ac": "paired_rainfed_return_above_operating_usd_ac",
        "return_above_total_usd_ac": "paired_rainfed_return_above_total_usd_ac",
    }
    baseline_keys = ["base_system", "crop_code", "year"]
    # Spatial datasets require a same-site rainfed baseline. State-level
    # datasets do not contain site_id and retain the original pairing.
    if "site_id" in df.columns:
        baseline_keys.append("site_id")
    rainfed = (
        df[df["water_regime"].str.casefold().eq("rainfed")]
        [[*baseline_keys, *baseline_metrics.keys()]]
        .rename(columns=baseline_metrics)
        .groupby(baseline_keys, as_index=False)
        .mean(numeric_only=True)
    )
    df = df.merge(
        rainfed,
        on=baseline_keys,
        how="left",
        validate="many_to_one",
    )

    df["incremental_iwue_kg_m3"] = np.where(
        irrig_mm > 0,
        (df["yield_kg_ha"] - df["paired_rainfed_yield_kg_ha"])
        / (irrig_mm * 10.0),
        np.nan,
    )

    df["marginal_gross_return_usd_ac"] = (
        df["gross_income_simulated_usd_ac"]
        - df["paired_rainfed_gross_income_usd_ac"]
    )
    df["marginal_return_above_operating_usd_ac"] = (
        df["return_above_operating_usd_ac"]
        - df["paired_rainfed_return_above_operating_usd_ac"]
    )
    df["marginal_return_above_total_usd_ac"] = (
        df["return_above_total_usd_ac"]
        - df["paired_rainfed_return_above_total_usd_ac"]
    )

    # A rainfed record is its own baseline, so its marginal value is zero.
    is_rainfed = df["water_regime"].str.casefold().eq("rainfed")
    for col in [
        "marginal_gross_return_usd_ac",
        "marginal_return_above_operating_usd_ac",
        "marginal_return_above_total_usd_ac",
    ]:
        df.loc[is_rainfed & df[col].notna(), col] = 0.0

    df["marginal_return_per_mm_usd_ac_mm"] = np.where(
        irrig_mm > 0,
        df["marginal_return_above_operating_usd_ac"] / irrig_mm,
        np.nan,
    )
    return df


def aggregate_system_year(df: pd.DataFrame) -> pd.DataFrame:
    """Create one state-average record per cropping system and year."""
    group_cols = [
        "cropping_system",
        "base_system",
        "rotation",
        "water_regime",
        "year",
    ]
    agg = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            yield_kg_ha=("yield_kg_ha", "mean"),
            yield_bu_ac=("yield_bu_ac", "mean"),
            irrigation_mm=("irrigation_mm", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            gross_income_simulated_usd_ac=("gross_income_simulated_usd_ac", "mean"),
            return_above_operating_usd_ac=("return_above_operating_usd_ac", "mean"),
            return_above_total_usd_ac=("return_above_total_usd_ac", "mean"),
            marginal_gross_return_usd_ac=("marginal_gross_return_usd_ac", "mean"),
            marginal_return_above_operating_usd_ac=("marginal_return_above_operating_usd_ac", "mean"),
            marginal_return_above_total_usd_ac=("marginal_return_above_total_usd_ac", "mean"),
            marginal_return_per_mm_usd_ac_mm=("marginal_return_per_mm_usd_ac_mm", "mean"),
            irrigation_productivity_kg_m3=("irrigation_productivity_kg_m3", "mean"),
            incremental_iwue_kg_m3=("incremental_iwue_kg_m3", "mean"),
            n_sites=("n_sites", "max"),
        )
    )
    return agg


def summarize_systems(annual: pd.DataFrame) -> pd.DataFrame:
    """Long-term per-acre totals, averages, and variability."""
    g = annual.groupby(
        ["cropping_system", "base_system", "rotation", "water_regime"],
        as_index=False,
        dropna=False,
    )
    summary = g.agg(
        years=("year", "nunique"),
        mean_yield_kg_ha=("yield_kg_ha", "mean"),
        yield_cv_pct=("yield_kg_ha", lambda x: 100.0 * x.std(ddof=1) / x.mean() if x.mean() else np.nan),
        mean_irrigation_mm=("irrigation_mm", "mean"),
        total_irrigation_mm=("irrigation_mm", "sum"),
        irrigation_productivity_kg_m3=("irrigation_productivity_kg_m3", "mean"),
        incremental_iwue_kg_m3=("incremental_iwue_kg_m3", "mean"),
        mean_gross_income_usd_ac=("gross_income_simulated_usd_ac", "mean"),
        total_gross_income_usd_ac=("gross_income_simulated_usd_ac", "sum"),
        mean_return_above_operating_usd_ac=("return_above_operating_usd_ac", "mean"),
        total_return_above_operating_usd_ac=("return_above_operating_usd_ac", "sum"),
        mean_return_above_total_usd_ac=("return_above_total_usd_ac", "mean"),
        total_return_above_total_usd_ac=("return_above_total_usd_ac", "sum"),
        mean_marginal_gross_return_usd_ac=("marginal_gross_return_usd_ac", "mean"),
        total_marginal_gross_return_usd_ac=("marginal_gross_return_usd_ac", "sum"),
        mean_marginal_return_above_operating_usd_ac=("marginal_return_above_operating_usd_ac", "mean"),
        total_marginal_return_above_operating_usd_ac=("marginal_return_above_operating_usd_ac", "sum"),
        mean_marginal_return_above_total_usd_ac=("marginal_return_above_total_usd_ac", "mean"),
        total_marginal_return_above_total_usd_ac=("marginal_return_above_total_usd_ac", "sum"),
        mean_marginal_return_per_mm_usd_ac_mm=("marginal_return_per_mm_usd_ac_mm", "mean"),
    )
    return summary


OBJECTIVES = {
    "Highest mean yield": ("mean_yield_kg_ha", False),
    "Highest irrigation water-use efficiency": ("incremental_iwue_kg_m3", False),
    "Highest irrigation productivity": ("irrigation_productivity_kg_m3", False),
    "Highest average gross income": ("mean_gross_income_usd_ac", False),
    "Highest cumulative gross income over common years": (
        "cumulative_gross_income_common_years_usd_ac",
        False,
    ),
    "Highest average return above operating cost": (
        "mean_return_above_operating_usd_ac",
        False,
    ),
    "Cumulative highest total return above operating cost over common years": (
        "cumulative_return_above_operating_common_years_usd_ac",
        False,
    ),
    "Highest average return above total cost": (
        "mean_return_above_total_usd_ac",
        False,
    ),
    "Highest marginal return above operating cost": (
        "mean_marginal_return_above_operating_usd_ac",
        False,
    ),
    "Highest marginal return per mm irrigation": (
        "mean_marginal_return_per_mm_usd_ac_mm",
        False,
    ),
    "Lowest average irrigation": ("mean_irrigation_mm", True),
    "Lowest yield variability": ("yield_cv_pct", True),
}


def rank_systems(
    summary: pd.DataFrame,
    objective_label: str,
    max_mean_irrigation_mm: float | None = None,
    min_mean_return_operating: float | None = None,
) -> pd.DataFrame:
    out = summary.copy()
    if max_mean_irrigation_mm is not None:
        # Unknown irrigation must not be treated as zero when a producer sets
        # a water limit. Rainfed records carry an explicit zero and remain eligible.
        out = out[
            out["mean_irrigation_mm"].notna()
            & (out["mean_irrigation_mm"] <= float(max_mean_irrigation_mm))
        ]
    if min_mean_return_operating is not None:
        out = out[
            out["mean_return_above_operating_usd_ac"]
            >= float(min_mean_return_operating)
        ]

    metric, ascending = OBJECTIVES[objective_label]
    out = out.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def aggregate_crop_year(df: pd.DataFrame) -> pd.DataFrame:
    """Create one state-average record per system, regime, crop, and year."""
    group_cols = [
        "cropping_system",
        "base_system",
        "rotation",
        "water_regime",
        "crop_code",
        "crop_name",
        "year",
    ]
    metric_map = {
        "yield_kg_ha": "mean",
        "yield_bu_ac": "mean",
        "irrigation_mm": "mean",
        "gross_income_simulated_usd_ac": "mean",
        "return_above_operating_usd_ac": "mean",
        "return_above_total_usd_ac": "mean",
        # Retain crop-level marginal measures so every Farm economics option
        # can be plotted in the annual crop charts.
        "marginal_gross_return_usd_ac": "mean",
        "marginal_return_above_operating_usd_ac": "mean",
        "marginal_return_above_total_usd_ac": "mean",
        "marginal_return_per_mm_usd_ac_mm": "mean",
        "irrigation_productivity_kg_m3": "mean",
        "incremental_iwue_kg_m3": "mean",
    }
    available = {col: func for col, func in metric_map.items() if col in df.columns}
    return (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(available)
        .sort_values(["base_system", "water_regime", "crop_code", "year"])
        .reset_index(drop=True)
    )


def add_crop_cumulative_metrics(
    crop_annual: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Add cumulative crop yield and returns, retaining flat years between crops.

    Rotation crops are not present every year. Missing crop-years inside an
    otherwise selected period therefore contribute zero and produce flat
    cumulative segments. Coverage tables should be consulted to distinguish
    scheduled crop absence from an incomplete source series.
    """
    if crop_annual.empty:
        return crop_annual.copy()

    start = int(start_year if start_year is not None else crop_annual["year"].min())
    end = int(end_year if end_year is not None else crop_annual["year"].max())
    years = pd.DataFrame({"year": range(start, end + 1)})
    keys = [
        "cropping_system",
        "base_system",
        "rotation",
        "water_regime",
        "crop_code",
        "crop_name",
    ]
    value_cols = [
        "yield_kg_ha",
        "gross_income_simulated_usd_ac",
        "return_above_operating_usd_ac",
    ]

    frames: list[pd.DataFrame] = []
    for key_values, group in crop_annual.groupby(keys, dropna=False, sort=False):
        expanded = years.merge(group[["year", *value_cols]], on="year", how="left")
        for key, value in zip(keys, key_values):
            expanded[key] = value
        expanded["crop_observed_this_year"] = expanded["yield_kg_ha"].notna()
        for col in value_cols:
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce").fillna(0.0)
        expanded["cumulative_yield_kg_ha"] = expanded["yield_kg_ha"].cumsum()
        expanded["cumulative_gross_return_usd_ac"] = expanded[
            "gross_income_simulated_usd_ac"
        ].cumsum()
        expanded["cumulative_return_after_operational_cost_usd_ac"] = expanded[
            "return_above_operating_usd_ac"
        ].cumsum()
        frames.append(expanded)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["base_system", "water_regime", "crop_code", "year"]
    )
