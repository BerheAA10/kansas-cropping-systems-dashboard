from pathlib import Path

import pandas as pd

from src.data_loader import apply_regime_relabels, load_simulation_data


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [1981, 1981, 1981],
            "cropping_system": [
                "SB-MZ-SG | Irrigated",
                "SB-MZ-SG | Potential",
                "SB-MZ | Irrigated",
            ],
            "base_system": ["SB-MZ-SG", "SB-MZ-SG", "SB-MZ"],
            "rotation": ["SB-MZ-SG", "SB-MZ-SG", "SB-MZ"],
            "crop_code": ["MZ", "MZ", "MZ"],
            "water_regime": ["Irrigated", "Potential", "Irrigated"],
            "yield_kg_ha": [4200.0, 11800.0, 9000.0],
            "irrigation_mm": [175.0, 0.0, 150.0],
            "rainfall_mm": [500.0, 500.0, 500.0],
            "n_sites": [2776, 2776, 2776],
        }
    )


def test_sb_mz_sg_irrigated_label_is_reclassified_as_rainfed():
    original = sample_rows()
    corrected = apply_regime_relabels(original)

    row = corrected.iloc[0]
    assert row["water_regime"] == "Rainfed"
    assert row["cropping_system"] == "SB-MZ-SG | Rainfed"
    assert row["irrigation_mm"] == 0.0
    assert row["yield_kg_ha"] == original.iloc[0]["yield_kg_ha"]


def test_potential_and_other_system_irrigated_rows_are_untouched():
    original = sample_rows()
    corrected = apply_regime_relabels(original)

    assert corrected.iloc[1]["water_regime"] == "Potential"
    assert corrected.iloc[1]["yield_kg_ha"] == original.iloc[1]["yield_kg_ha"]
    assert corrected.iloc[2]["water_regime"] == "Irrigated"
    assert corrected.iloc[2]["irrigation_mm"] == original.iloc[2]["irrigation_mm"]


def test_portable_loader_applies_regime_correction(tmp_path: Path):
    source = tmp_path / "simulation.csv"
    sample_rows().to_csv(source, index=False)

    loaded = load_simulation_data(source)
    sb_mz_sg_regimes = set(
        loaded.loc[loaded["base_system"].eq("SB-MZ-SG"), "water_regime"]
    )
    assert sb_mz_sg_regimes == {"Rainfed", "Potential"}
    assert not (
        loaded["base_system"].eq("SB-MZ-SG")
        & loaded["water_regime"].eq("Irrigated")
    ).any()
