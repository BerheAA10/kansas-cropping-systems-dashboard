import numpy as np
import pandas as pd

from src.spatial import (
    build_complete_master_grid,
    compact_money,
    grid_heatmap_arrays,
    idw_fill_complete_grid,
    robust_color_range,
)


def test_compact_money_preserves_scale():
    assert compact_money(700) == "$700"
    assert compact_money(7000) == "$7K"
    assert compact_money(1_250_000) == "$1.2M"


def test_idw_fills_every_master_site():
    master = pd.DataFrame(
        {
            "site_id": ["a", "b", "c"],
            "latitude": [38.0, 38.1, 38.2],
            "longitude": [-98.0, -98.0, -98.0],
        }
    )
    observed = pd.DataFrame({"site_id": ["a", "c"], "value": [10.0, 30.0]})
    result = idw_fill_complete_grid(master, observed, "value", neighbors=2)
    assert len(result) == 3
    assert result["value"].notna().all()
    assert np.isfinite(result["value"]).all()
    assert result.loc[result["site_id"].eq("b"), "filled_by_idw"].item()
    assert 10.0 < result.loc[result["site_id"].eq("b"), "value"].item() < 30.0


def test_master_grid_restores_internal_lattice_holes():
    observed_sites = pd.DataFrame(
        {
            "site_id": ["a", "c", "d", "f", "g", "h", "i"],
            "latitude": [38.0, 38.0, 38.1, 38.1, 38.2, 38.2, 38.2],
            "longitude": [-98.0, -97.8, -98.0, -97.8, -98.0, -97.9, -97.8],
        }
    )
    grid = build_complete_master_grid(observed_sites, coordinate_precision=1)
    # The complete third row establishes the 0.1-degree lattice spacing.
    assert len(grid) == 9
    assert set(grid.groupby("latitude").size()) == {3}
    assert grid["grid_site_status"].eq("restored_grid_site").sum() == 2


def test_heatmap_has_no_internal_missing_cells_after_fill():
    master = pd.DataFrame(
        {
            "site_id": ["a", "b", "c", "d", "e", "f"],
            "latitude": [38.0, 38.0, 38.0, 38.1, 38.1, 38.1],
            "longitude": [-98.0, -97.9, -97.8, -98.0, -97.9, -97.8],
        }
    )
    observed = pd.DataFrame({"site_id": ["a", "c", "d", "f"], "value": [1, 3, 4, 6]})
    filled = idw_fill_complete_grid(master, observed, "value")
    _, _, z, _ = grid_heatmap_arrays(filled, "value", coordinate_precision=1)
    assert z.shape == (2, 3)
    assert np.isfinite(z).all()


def test_robust_range_returns_none_for_constant_values():
    assert robust_color_range(pd.Series([5.0, 5.0, 5.0])) is None


def test_master_grid_ignores_cross_file_coordinate_rounding_noise():
    observed_sites = pd.DataFrame(
        {
            "site_id": [
                "37_0417N_094_6250W",
                "37_0417N_094_7083W",
                "37_1250N_094_6250W",
                "37_1250N_094_7083W",
                # Duplicate sites from another export rounded to three decimals.
                "37_0417N_094_6250W",
                "37_1250N_094_7083W",
            ],
            "latitude": [37.0417, 37.0417, 37.1250, 37.1250, 37.042, 37.125],
            "longitude": [-94.6250, -94.7083, -94.6250, -94.7083, -94.625, -94.708],
        }
    )
    grid = build_complete_master_grid(observed_sites)
    assert len(grid) == 4
    assert grid["latitude"].nunique() == 2
    assert grid["longitude"].nunique() == 2
    assert len(grid) < 100


def test_authoritative_master_grid_keeps_only_original_sites():
    from src.spatial import build_authoritative_master_grid

    observed_sites = pd.DataFrame(
        {
            "site_id": [
                "37_0417N_094_6250W",
                "37_0417N_094_7083W",
                "37_1250N_094_6250W",
                "37_1250N_094_7083W",
                # Duplicate exports with rounded coordinates.
                "37_0417N_094_6250W",
            ],
            "latitude": [37.0417, 37.0417, 37.1250, 37.1250, 37.042],
            "longitude": [-94.6250, -94.7083, -94.6250, -94.7083, -94.625],
        }
    )
    grid = build_authoritative_master_grid(observed_sites)
    assert len(grid) == 4
    assert grid["site_id"].nunique() == 4
    assert grid["grid_site_status"].eq("observed_grid_site").all()


def test_idw_uses_coordinate_fallback_before_interpolation():
    master = pd.DataFrame(
        {
            "site_id": ["canonical-a", "canonical-b"],
            "latitude": [37.0417, 37.1250],
            "longitude": [-94.6250, -94.6250],
        }
    )
    observed = pd.DataFrame(
        {
            "site_id": ["different-a", "different-b"],
            "latitude": [37.042, 37.125],
            "longitude": [-94.625, -94.625],
            "value": [10.0, 20.0],
        }
    )
    result = idw_fill_complete_grid(master, observed, "value")
    assert result["fill_method"].eq("observed").all()
    assert result["filled_by_idw"].eq(False).all()
    assert result["value"].tolist() == [10.0, 20.0]


def test_complete_authoritative_site_map_uses_no_interpolation():
    site_count = 2776
    master = pd.DataFrame(
        {
            "site_id": [f"site_{index:04d}" for index in range(site_count)],
            "latitude": 37.0 + (np.arange(site_count) // 60) * 0.0833,
            "longitude": -102.0 + (np.arange(site_count) % 60) * 0.0833,
        }
    )
    observed = pd.DataFrame(
        {
            "site_id": master["site_id"],
            "value": np.linspace(100.0, 500.0, site_count),
        }
    )
    result = idw_fill_complete_grid(master, observed, "value")
    assert len(result) == site_count
    assert result["fill_method"].eq("observed").all()
    assert result["filled_by_idw"].eq(False).all()
