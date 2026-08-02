from __future__ import annotations

import numpy as np
import pandas as pd


def compact_money(value: float | int | None, suffix: str = "") -> str:
    """Format USD values compactly without changing the underlying value."""
    if value is None or pd.isna(value):
        return "Not available"
    number = float(value)
    magnitude = abs(number)
    sign = "-" if number < 0 else ""
    number = abs(number)
    if magnitude >= 1_000_000_000:
        text = f"{number / 1_000_000_000:.1f}B"
    elif magnitude >= 1_000_000:
        text = f"{number / 1_000_000:.1f}M"
    elif magnitude >= 1_000:
        text = f"{number / 1_000:.1f}K"
    else:
        text = f"{number:,.0f}"
    text = text.replace(".0K", "K").replace(".0M", "M").replace(".0B", "B")
    return f"{sign}${text}{suffix}"


def _site_id_coordinates(site_id: object) -> tuple[float | None, float | None]:
    """Parse canonical DSSAT coordinates such as 37_0417N_094_6250W."""
    import re

    match = re.search(
        r"(?P<latdeg>\d{1,2})[_\.](?P<latdec>\d{3,6})(?P<lathem>[NS]).*?"
        r"(?P<londeg>\d{1,3})[_\.](?P<londec>\d{3,6})(?P<lonhem>[EW])",
        str(site_id),
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    lat = float(f"{match.group('latdeg')}.{match.group('latdec')}")
    lon = float(f"{match.group('londeg')}.{match.group('londec')}")
    if match.group("lathem").upper() == "S":
        lat = -lat
    if match.group("lonhem").upper() == "W":
        lon = -lon
    return lat, lon


def _infer_step(
    values: pd.Series,
    fallback: float = 0.0833,
    *,
    min_step: float = 0.02,
    max_step: float = 0.20,
) -> float:
    """Infer regular grid spacing while ignoring coordinate noise.

    Multiple source exports can carry the same site at slightly different
    rounded coordinates (for example 37.0417 and 37.042). Tiny differences
    must not be interpreted as the statewide lattice spacing.
    """
    unique = np.sort(
        pd.to_numeric(values, errors="coerce").dropna().round(4).unique()
    )
    if len(unique) < 2:
        return float(fallback)
    diffs = np.diff(unique)
    diffs = diffs[
        np.isfinite(diffs)
        & (diffs >= float(min_step))
        & (diffs <= float(max_step))
    ]
    if not len(diffs):
        return float(fallback)

    # Missing sites can produce multiples of the true spacing. Start from the
    # lower part of the distribution, then take the median of its tight cluster.
    # This preserves a 0.0833-degree lattice even when rounded coordinates also
    # produce occasional 0.0834-degree differences.
    seed = float(np.quantile(diffs, 0.20))
    cluster = diffs[(diffs >= seed * 0.80) & (diffs <= seed * 1.20)]
    step = float(np.median(cluster if len(cluster) else diffs))
    step = round(step, 4)
    if not np.isfinite(step) or step < min_step or step > max_step:
        return float(fallback)
    return step


def _make_generated_site_id(latitude: float, longitude: float) -> str:
    lat_hemi = "N" if latitude >= 0 else "S"
    lon_hemi = "E" if longitude >= 0 else "W"
    lat_abs = abs(latitude)
    lon_abs = abs(longitude)
    lat_deg = int(lat_abs)
    lon_deg = int(lon_abs)
    lat_dec = int(round((lat_abs - lat_deg) * 10000))
    lon_dec = int(round((lon_abs - lon_deg) * 10000))
    return f"{lat_deg:02d}_{lat_dec:04d}{lat_hemi}_{lon_deg:03d}_{lon_dec:04d}{lon_hemi}"


def build_complete_master_grid(
    observed_sites: pd.DataFrame,
    *,
    coordinate_precision: int = 4,
    max_grid_sites: int = 10_000,
) -> pd.DataFrame:
    """Preserve every observed site and add only genuinely missing grid cells.

    The 2,776 DSSAT site IDs are authoritative. A synthetic display lattice is
    used only to identify internal gaps; it must never replace or relocate an
    observed site. This guarantees that complete site-year data are counted as
    direct observations before any IDW filling is applied.
    """
    required = {"site_id", "latitude", "longitude"}
    missing = required.difference(observed_sites.columns)
    if missing:
        raise ValueError(
            "Observed site table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    sites = observed_sites[["site_id", "latitude", "longitude"]].copy()
    sites["site_id"] = sites["site_id"].astype(str).str.strip()
    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")

    parsed = sites["site_id"].map(_site_id_coordinates)
    parsed_lat = pd.Series([item[0] for item in parsed], index=sites.index, dtype=float)
    parsed_lon = pd.Series([item[1] for item in parsed], index=sites.index, dtype=float)
    sites.loc[parsed_lat.notna(), "latitude"] = parsed_lat[parsed_lat.notna()]
    sites.loc[parsed_lon.notna(), "longitude"] = parsed_lon[parsed_lon.notna()]

    sites = sites.dropna(subset=["site_id", "latitude", "longitude"])
    sites = sites[
        sites["latitude"].between(36.0, 41.5, inclusive="both")
        & sites["longitude"].between(-103.5, -93.0, inclusive="both")
    ].copy()
    if sites.empty:
        return pd.DataFrame(
            columns=["site_id", "latitude", "longitude", "grid_site_status"]
        )

    sites["latitude"] = sites["latitude"].round(coordinate_precision)
    sites["longitude"] = sites["longitude"].round(coordinate_precision)

    # Preserve one canonical coordinate pair for every original simulation ID.
    # Do not collapse the observed sites onto a synthetic common lattice.
    observed_exact = (
        sites.drop_duplicates(subset=["site_id"], keep="first")
        .sort_values(["latitude", "longitude", "site_id"])
        .reset_index(drop=True)
    )
    observed_exact["grid_site_status"] = "observed_grid_site"

    # Coordinate-unique copy is used only to infer the underlying lattice.
    lattice_source = (
        observed_exact.drop_duplicates(subset=["latitude", "longitude"], keep="first")
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )
    lon_step = _infer_step(lattice_source["longitude"])
    lat_step = _infer_step(lattice_source["latitude"])
    observed_bounds = (
        lattice_source.groupby("latitude", as_index=False)
        .agg(west=("longitude", "min"), east=("longitude", "max"))
        .sort_values("latitude")
    )
    lat_min = float(observed_bounds["latitude"].min())
    lat_max = float(observed_bounds["latitude"].max())
    lat_count = int(round((lat_max - lat_min) / lat_step)) + 1
    if lat_count <= 0 or lat_count > 100:
        raise ValueError(
            f"Unsafe latitude-grid reconstruction: step={lat_step}, rows={lat_count}."
        )

    expected_lats = np.round(
        lat_min + np.arange(lat_count, dtype=float) * lat_step,
        coordinate_precision,
    )
    row_lat = observed_bounds["latitude"].to_numpy(float)
    row_west = observed_bounds["west"].to_numpy(float)
    row_east = observed_bounds["east"].to_numpy(float)
    expected_east = np.interp(expected_lats, row_lat, row_east)
    global_west = round(float(np.nanmin(row_west)), coordinate_precision)

    rows: list[pd.DataFrame] = []
    total_expected = 0
    for latitude, east_raw in zip(expected_lats, expected_east):
        west = global_west
        east = round(float(east_raw), coordinate_precision)
        count = int(np.floor((east - west) / lon_step + 0.5)) + 1
        if count <= 0 or count > 250:
            raise ValueError(
                f"Unsafe longitude-grid reconstruction at latitude {latitude}: "
                f"step={lon_step}, columns={count}."
            )
        total_expected += count
        if total_expected > int(max_grid_sites):
            raise ValueError(
                f"Unsafe candidate-grid size ({total_expected:,} sites); "
                f"expected no more than {int(max_grid_sites):,}."
            )
        longitudes = np.round(
            west + np.arange(count, dtype=float) * lon_step,
            coordinate_precision,
        )
        rows.append(
            pd.DataFrame(
                {
                    "latitude": np.repeat(float(latitude), count),
                    "longitude": longitudes,
                }
            )
        )

    candidates = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(["latitude", "longitude"])
        .reset_index(drop=True)
    )

    # Remove candidate cells already represented by a nearby actual site. The
    # tolerance handles row-phase and rounding differences without dropping true
    # holes one full grid step away.
    mean_lat_rad = np.deg2rad(float(observed_exact["latitude"].mean()))
    lon_scale = float(np.cos(mean_lat_rad))
    observed_xy = np.column_stack(
        [
            observed_exact["latitude"].to_numpy(float),
            observed_exact["longitude"].to_numpy(float) * lon_scale,
        ]
    )
    candidate_xy = np.column_stack(
        [
            candidates["latitude"].to_numpy(float),
            candidates["longitude"].to_numpy(float) * lon_scale,
        ]
    )
    nearest_sq = np.empty(len(candidate_xy), dtype=float)
    chunk_size = 256
    for start_idx in range(0, len(candidate_xy), chunk_size):
        stop_idx = min(start_idx + chunk_size, len(candidate_xy))
        block = candidate_xy[start_idx:stop_idx]
        distances_sq = (
            (block[:, None, 0] - observed_xy[None, :, 0]) ** 2
            + (block[:, None, 1] - observed_xy[None, :, 1]) ** 2
        )
        nearest_sq[start_idx:stop_idx] = distances_sq.min(axis=1)

    spacing = min(float(lat_step), float(lon_step) * lon_scale)
    proximity_tolerance = max(0.005, spacing * 0.38)
    generated = candidates.loc[nearest_sq > proximity_tolerance**2].copy()
    generated["site_id"] = [
        "FILL_" + _make_generated_site_id(lat, lon)
        for lat, lon in zip(generated["latitude"], generated["longitude"])
    ]
    generated["grid_site_status"] = "restored_grid_site"

    result = pd.concat(
        [
            observed_exact[["site_id", "latitude", "longitude", "grid_site_status"]],
            generated[["site_id", "latitude", "longitude", "grid_site_status"]],
        ],
        ignore_index=True,
    )
    if len(result) > int(max_grid_sites):
        raise ValueError(
            f"Unsafe completed-grid size ({len(result):,} sites); "
            f"expected no more than {int(max_grid_sites):,}."
        )
    return (
        result.drop_duplicates(subset=["site_id"], keep="first")
        .sort_values(["latitude", "longitude", "grid_site_status", "site_id"])
        .reset_index(drop=True)
    )

def build_authoritative_master_grid(
    observed_sites: pd.DataFrame,
    *,
    coordinate_precision: int = 4,
) -> pd.DataFrame:
    """Return the exact simulation-site grid without adding artificial cells.

    The canonical DSSAT site ID is used to correct cross-file coordinate rounding.
    One row is retained for each original simulation site.
    """
    required = {"site_id", "latitude", "longitude"}
    missing = required.difference(observed_sites.columns)
    if missing:
        raise ValueError(
            "Observed site table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    sites = observed_sites[["site_id", "latitude", "longitude"]].copy()
    sites["site_id"] = sites["site_id"].astype(str).str.strip()
    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")

    parsed = sites["site_id"].map(_site_id_coordinates)
    parsed_lat = pd.Series([item[0] for item in parsed], index=sites.index, dtype=float)
    parsed_lon = pd.Series([item[1] for item in parsed], index=sites.index, dtype=float)
    sites.loc[parsed_lat.notna(), "latitude"] = parsed_lat[parsed_lat.notna()]
    sites.loc[parsed_lon.notna(), "longitude"] = parsed_lon[parsed_lon.notna()]

    sites = sites.dropna(subset=["site_id", "latitude", "longitude"])
    sites = sites[
        sites["latitude"].between(36.0, 41.5, inclusive="both")
        & sites["longitude"].between(-103.5, -93.0, inclusive="both")
    ].copy()
    sites["latitude"] = sites["latitude"].round(coordinate_precision)
    sites["longitude"] = sites["longitude"].round(coordinate_precision)

    # One authoritative coordinate pair per original site ID.
    sites = (
        sites.groupby("site_id", as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .sort_values(["latitude", "longitude", "site_id"])
        .reset_index(drop=True)
    )
    sites["grid_site_status"] = "observed_grid_site"
    return sites[["site_id", "latitude", "longitude", "grid_site_status"]]

def idw_fill_complete_grid(
    master_sites: pd.DataFrame,
    observed: pd.DataFrame,
    value_col: str,
    *,
    neighbors: int = 12,
    power: float = 2.0,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Return every master-grid site with a finite observed or IDW value.

    Missing values are estimated from nearest observed sites. The function never
    returns blank values when at least one numeric observation exists. A global
    observed mean is used only as a final numerical safeguard.
    """
    required_master = {"site_id", "latitude", "longitude"}
    missing_master = required_master.difference(master_sites.columns)
    if missing_master:
        raise ValueError(
            "Master site grid is missing required columns: "
            + ", ".join(sorted(missing_master))
        )
    if value_col not in observed.columns or "site_id" not in observed.columns:
        return pd.DataFrame(columns=["site_id", "latitude", "longitude", value_col])

    extra_master = [c for c in ["grid_site_status"] if c in master_sites.columns]
    grid = master_sites[["site_id", "latitude", "longitude", *extra_master]].copy()
    grid["latitude"] = pd.to_numeric(grid["latitude"], errors="coerce")
    grid["longitude"] = pd.to_numeric(grid["longitude"], errors="coerce")
    grid = (
        grid.dropna(subset=["site_id", "latitude", "longitude"])
        .drop_duplicates(subset=["site_id"], keep="first")
        .reset_index(drop=True)
    )

    # Match direct values by normalized site ID first, then use canonical
    # coordinates as a fallback. This prevents cross-file site-ID formatting or
    # coordinate-rounding differences from turning valid sites into IDW fills.
    grid["site_id"] = grid["site_id"].astype(str).str.strip()
    grid["_site_key"] = grid["site_id"].str.casefold()

    observed_columns = ["site_id", value_col]
    for coordinate in ["latitude", "longitude"]:
        if coordinate in observed.columns:
            observed_columns.append(coordinate)
    values = observed[observed_columns].copy()
    values["site_id"] = values["site_id"].astype(str).str.strip()
    values["_site_key"] = values["site_id"].str.casefold()
    values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
    values_by_site = values.groupby("_site_key", as_index=False)[value_col].mean()
    result = grid.merge(values_by_site, on="_site_key", how="left")

    # Coordinate fallback for any unmatched site IDs. Site IDs are parsed first
    # because they preserve the canonical four-decimal DSSAT coordinates.
    if {"latitude", "longitude"}.issubset(values.columns):
        value_coords = values[["site_id", "latitude", "longitude", value_col]].copy()
        value_coords["latitude"] = pd.to_numeric(value_coords["latitude"], errors="coerce")
        value_coords["longitude"] = pd.to_numeric(value_coords["longitude"], errors="coerce")
        parsed = value_coords["site_id"].map(_site_id_coordinates)
        parsed_lat = pd.Series([item[0] for item in parsed], index=value_coords.index, dtype=float)
        parsed_lon = pd.Series([item[1] for item in parsed], index=value_coords.index, dtype=float)
        value_coords.loc[parsed_lat.notna(), "latitude"] = parsed_lat[parsed_lat.notna()]
        value_coords.loc[parsed_lon.notna(), "longitude"] = parsed_lon[parsed_lon.notna()]
        value_coords["_lat_key"] = value_coords["latitude"].round(3)
        value_coords["_lon_key"] = value_coords["longitude"].round(3)
        values_by_coord = (
            value_coords.dropna(subset=["_lat_key", "_lon_key", value_col])
            .groupby(["_lat_key", "_lon_key"], as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "_coordinate_value"})
        )
        result["_lat_key"] = result["latitude"].round(3)
        result["_lon_key"] = result["longitude"].round(3)
        result = result.merge(values_by_coord, on=["_lat_key", "_lon_key"], how="left")
        unmatched = result[value_col].isna()
        result.loc[unmatched, value_col] = result.loc[unmatched, "_coordinate_value"]

    known_mask = result[value_col].notna() & np.isfinite(result[value_col])
    if not known_mask.any():
        return result.iloc[0:0].drop(columns=[c for c in result.columns if c.startswith("_")], errors="ignore").copy()

    result["fill_method"] = np.where(known_mask, "observed", "idw")
    missing_mask = ~known_mask
    if not missing_mask.any():
        result["filled_by_idw"] = False
        return result.drop(columns=[c for c in result.columns if c.startswith("_")], errors="ignore")

    known = result.loc[known_mask, ["latitude", "longitude", value_col]].copy()
    unknown = result.loc[missing_mask, ["latitude", "longitude"]].copy()

    mean_lat_rad = np.deg2rad(result["latitude"].mean())
    lon_scale = np.cos(mean_lat_rad)
    known_xy = np.column_stack(
        [known["latitude"].to_numpy(float), known["longitude"].to_numpy(float) * lon_scale]
    )
    unknown_xy = np.column_stack(
        [unknown["latitude"].to_numpy(float), unknown["longitude"].to_numpy(float) * lon_scale]
    )
    known_values = known[value_col].to_numpy(float)
    k = max(1, min(int(neighbors), len(known_values)))
    filled = np.empty(len(unknown_xy), dtype=float)

    for start in range(0, len(unknown_xy), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(unknown_xy))
        block = unknown_xy[start:stop]
        distances_sq = (
            (block[:, None, 0] - known_xy[None, :, 0]) ** 2
            + (block[:, None, 1] - known_xy[None, :, 1]) ** 2
        )
        nearest_idx = np.argpartition(distances_sq, kth=k - 1, axis=1)[:, :k]
        nearest_dist_sq = np.take_along_axis(distances_sq, nearest_idx, axis=1)
        nearest_values = known_values[nearest_idx]
        weights = 1.0 / np.maximum(nearest_dist_sq, 1e-20) ** (power / 2.0)
        estimates = np.sum(weights * nearest_values, axis=1) / np.sum(weights, axis=1)
        filled[start:stop] = estimates

    result.loc[missing_mask, value_col] = filled
    # Numerical safeguard: no blanks are allowed in the completed map.
    fallback = float(np.nanmean(known_values))
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(fallback)
    result["filled_by_idw"] = result["fill_method"].eq("idw")
    return result.drop(columns=[c for c in result.columns if c.startswith("_")], errors="ignore")


def grid_heatmap_arrays(
    filled_grid: pd.DataFrame,
    value_col: str,
    *,
    coordinate_precision: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a complete site grid to contiguous heatmap arrays.

    Returns longitudes, latitudes, values, and site IDs. Cells outside each
    supported Kansas row remain masked, while every internal grid cell is filled.
    """
    required = {"site_id", "latitude", "longitude", value_col}
    missing = required.difference(filled_grid.columns)
    if missing:
        raise ValueError("Filled grid is missing: " + ", ".join(sorted(missing)))

    data = filled_grid[list(required)].copy()
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce").round(coordinate_precision)
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce").round(coordinate_precision)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=["latitude", "longitude", value_col])
    lats = np.sort(data["latitude"].unique())
    lons = np.sort(data["longitude"].unique())
    z = data.pivot_table(index="latitude", columns="longitude", values=value_col, aggfunc="mean").reindex(index=lats, columns=lons)
    ids = data.pivot_table(index="latitude", columns="longitude", values="site_id", aggfunc="first").reindex(index=lats, columns=lons)
    return lons, lats, z.to_numpy(float), ids.to_numpy(object)


def robust_color_range(values: pd.Series, lower: float = 0.02, upper: float = 0.98):
    """Return a robust Plotly color range while preserving constant fields."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    low = float(numeric.quantile(lower))
    high = float(numeric.quantile(upper))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None
    return (low, high)
