from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

REQUIRED_SIM_COLUMNS = {
    "year",
    "cropping_system",
    "base_system",
    "rotation",
    "crop_code",
    "water_regime",
    "yield_kg_ha",
    "irrigation_mm",
}

CROP_NAMES = {
    "MZ": "Maize",
    "WT": "Wheat",
    "SG": "Sorghum",
    "SB": "Soybean",
}


def normalize_base_system(value: object) -> str:
    """Normalize continuous-crop names while preserving supported rotations."""
    text = str(value).strip().upper().replace("–", "-").replace("—", "-")
    compact = " ".join(text.replace("_", " ").replace("-", " ").split())
    rotation_tokens = {
        "SB MZ SG WT": "SB-MZ-SG-WT",
        "SB MZ SG": "SB-MZ-SG",
        "SB MZ": "SB-MZ",
    }
    if compact in rotation_tokens:
        return rotation_tokens[compact]

    continuous_aliases = {
        "MZ": "MZ",
        "MAIZE": "MZ",
        "CORN": "MZ",
        "CONTINUOUS MZ": "MZ",
        "CONTINUOUS MAIZE": "MZ",
        "CONTINUOUS CORN": "MZ",
        "SG": "SG",
        "SORGHUM": "SG",
        "CONTINUOUS SG": "SG",
        "CONTINUOUS SORGHUM": "SG",
        "WT": "WT",
        "WHEAT": "WT",
        "CONTINUOUS WT": "WT",
        "CONTINUOUS WHEAT": "WT",
        "SB": "SB",
        "SOYBEAN": "SB",
        "SOY": "SB",
        "CONTINUOUS SB": "SB",
        "CONTINUOUS SOYBEAN": "SB",
    }
    return continuous_aliases.get(compact, text)


def load_simulation_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load and validate the portable, state-level simulation dataset."""
    df = pd.read_csv(source)
    missing = REQUIRED_SIM_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Simulation dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for col in ["yield_kg_ha", "irrigation_mm", "rainfall_mm", "n_sites"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["crop_code", "water_regime", "cropping_system", "base_system", "rotation"]:
        out[col] = out[col].astype(str).str.strip()

    out["crop_code"] = out["crop_code"].str.upper()
    out["base_system"] = out["base_system"].map(normalize_base_system)
    out["rotation"] = out["rotation"].map(normalize_base_system)
    out["cropping_system"] = out["base_system"] + " | " + out["water_regime"]
    out["crop_name"] = out["crop_code"].map(CROP_NAMES).fillna(out["crop_code"])
    out = out[out["year"].between(1981, 2018, inclusive="both")]
    out = out[out["crop_code"].isin(CROP_NAMES)]
    return out.reset_index(drop=True)


def load_economic_data(path: str | Path) -> pd.DataFrame:
    """Load the normalized annual crop price and cost series."""
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    numeric = [
        "gross_value_survey_usd_ac",
        "operating_cost_usd_ac",
        "return_above_operating_survey_usd_ac",
        "total_production_cost_usd_ac",
        "return_above_total_survey_usd_ac",
        "price_usd_bu",
        "survey_yield_bu_ac",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["crop_code"] = df["crop_code"].astype(str).str.upper().str.strip()
    return df


def load_spatial_simulation_data(source: str | Path) -> pd.DataFrame:
    """Load the site-year spatial simulation dataset produced by the converter."""
    path = Path(source)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    required = {
        "year", "base_system", "rotation", "crop_code", "water_regime",
        "site_id", "latitude", "longitude", "yield_kg_ha", "irrigation_mm",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Spatial simulation dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for col in ["latitude", "longitude", "yield_kg_ha", "irrigation_mm", "rainfall_mm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["crop_code"] = out["crop_code"].astype(str).str.upper().str.strip()
    out["base_system"] = out["base_system"].map(normalize_base_system)
    out["rotation"] = out["rotation"].map(normalize_base_system)
    out["water_regime"] = out["water_regime"].astype(str).str.strip()
    out["cropping_system"] = out["base_system"] + " | " + out["water_regime"]
    out["crop_name"] = out["crop_code"].map(CROP_NAMES).fillna(out["crop_code"])
    out = out[
        out["year"].between(1981, 2018, inclusive="both")
        & out["crop_code"].isin(CROP_NAMES)
        & out["latitude"].between(36.0, 41.5, inclusive="both")
        & out["longitude"].between(-103.5, -93.0, inclusive="both")
    ]
    return out.reset_index(drop=True)
