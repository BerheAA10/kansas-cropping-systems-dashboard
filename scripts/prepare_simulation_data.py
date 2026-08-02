from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import apply_regime_relabels
from src.spatial import build_authoritative_master_grid

EXPECTED_SYSTEMS = ["MZ", "SG", "WT", "SB", "SB-MZ", "SB-MZ-SG", "SB-MZ-SG-WT"]
EXPECTED_REGIMES = ["Rainfed", "Irrigated", "Potential"]
VALID_CROPS = ["MZ", "WT", "SG", "SB"]

# Excel slicer metadata are irrelevant to values used by this workflow.
warnings.filterwarnings(
    "ignore",
    message="Slicer List extension is not supported and will be removed",
    category=UserWarning,
    module=r"openpyxl\.worksheet\._reader",
)

YEAR_ALIASES = [
    "rotation_year", "year", "yr", "crop_year", "cropyear", "harvest_year",
    "season_year", "simulation_year", "sim_year", "weather_year", "yyyy",
    "calendar_year",
]
YIELD_ALIASES = [
    "yield_kg_ha", "yield_kgha", "grain_yield_kg_ha", "grainyield_kg_ha",
    "mean_yield_kg_ha", "state_mean_yield_kg_ha", "average_yield_kg_ha",
    "avg_yield_kg_ha", "hwam_kg_ha", "mean_hwam_kg_ha", "mean_hwam",
    "hwam", "hwah", "yield", "mean_yield", "harvested_yield",
]
IRRIGATION_ALIASES = [
    "irrigation_mm", "irrigation_applied_mm", "applied_irrigation_mm",
    "seasonal_irrigation_mm", "total_irrigation_mm", "net_irrigation_mm",
    "irrigation_water_mm", "irrigation_amount_mm", "net_irrigation",
    "net_mm", "irrig_mm", "ircm", "irrig", "irrigation",
]
RAINFALL_ALIASES = [
    "rainfall_mm", "seasonal_rainfall_mm", "precipitation_mm", "rain_mm_prcm",
    "rain_mm", "rain", "prcp_mm", "season_prcp_mm", "prcm",
]
CROP_ALIASES = ["crop_code", "crop", "cr", "crop_name", "species"]
SITE_ALIASES = [
    "site_id", "site", "site_name", "grid_id", "location_id", "cell_id",
]
LATITUDE_ALIASES = [
    "site_latitude_from_folder", "site_latitude", "latitude", "xlat", "lat",
]
LONGITUDE_ALIASES = [
    "site_longitude_from_folder", "site_longitude", "longitude", "long", "lon",
]
SYSTEM_ALIASES = [
    "cropping_system", "base_system", "rotation", "rotation_name", "scenario",
    "cropping_sequence",
]
REGIME_ALIASES = [
    "water_regime", "production_system", "water_system", "treatment",
    "water_treatment", "irrigation_treatment",
]

CROP_PATTERNS = {
    "MZ": [r"\bmz\b", r"maize", r"\bcorn\b"],
    "WT": [r"\bwt\b", r"wheat"],
    "SG": [r"\bsg\b", r"sorghum"],
    "SB": [r"\bsb\b", r"soybean", r"\bsoy\b"],
}

METHOD_REFERENCE_PATTERN = re.compile(
    r"\b(?:exact|same|using|from|based\s+on)?\s*"
    r"(?:mz|maize|corn|wt|wheat|sg|sorghum|sb|soybean|soy)\s+"
    r"(?:method|template|workflow|reference|approach)\b",
    flags=re.IGNORECASE,
)

ROTATION_PATTERNS = [
    ("SB-MZ-SG-WT", [
        r"\bsb[\s_-]*mz[\s_-]*sg[\s_-]*wt\b",
        r"soybean[\s_-]*maize[\s_-]*sorghum[\s_-]*wheat",
        r"soybean[\s_-]*corn[\s_-]*sorghum[\s_-]*wheat",
    ]),
    ("SB-MZ-SG", [
        r"\bsb[\s_-]*mz[\s_-]*sg\b",
        r"soybean[\s_-]*maize[\s_-]*sorghum",
        r"soybean[\s_-]*corn[\s_-]*sorghum",
    ]),
    ("SB-MZ", [
        r"\bsb[\s_-]*mz\b",
        r"soybean[\s_-]*maize",
        r"soybean[\s_-]*corn",
    ]),
]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def select_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized = {normalize_name(c): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for alias in aliases:
        matches = [original for norm, original in normalized.items() if alias in norm]
        if len(matches) == 1:
            return matches[0]
    return None




def parse_site_coordinates(value: object) -> tuple[float | None, float | None]:
    """Parse site IDs such as 37_0417N_094_6250W."""
    text = str(value)
    match = re.search(
        r"(?P<latdeg>\d{1,2})[_\.](?P<latdec>\d+)\s*(?P<lathem>[NS]).*?"
        r"(?P<londeg>\d{1,3})[_\.](?P<londec>\d+)\s*(?P<lonhem>[EW])",
        text,
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


def crop_hits(text: str) -> list[str]:
    cleaned = METHOD_REFERENCE_PATTERN.sub(" ", str(text))
    low = " " + re.sub(r"[_\-.\\/]+", " ", cleaned.lower()) + " "
    hits: list[str] = []
    for code, patterns in CROP_PATTERNS.items():
        if any(re.search(pattern, low) for pattern in patterns):
            hits.append(code)
    return hits


def infer_crop(text: str) -> str | None:
    segments = [s for s in re.split(r"[\\/]+", str(text)) if s]
    for segment in reversed(segments):
        hits = crop_hits(segment)
        if len(hits) == 1:
            return hits[0]
    hits = crop_hits(str(text))
    return hits[0] if len(hits) == 1 else None


def normalize_crop_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    direct = {
        "MZ": "MZ", "MAIZE": "MZ", "CORN": "MZ",
        "WT": "WT", "WHEAT": "WT",
        "SG": "SG", "SORGHUM": "SG",
        "SB": "SB", "SOYBEAN": "SB", "SOY": "SB",
    }
    return direct.get(text.upper()) or infer_crop(text)


def explicit_rotation(text: str, require_rotation_context: bool) -> str | None:
    cleaned = METHOD_REFERENCE_PATTERN.sub(" ", str(text)).lower()
    normalized = re.sub(r"[^a-z0-9_-]+", " ", cleaned)
    context_ok = (
        not require_rotation_context
        or "rotation" in normalized
        or "sequence" in normalized
        or bool(re.search(r"\bsb[\s_-]*mz", normalized))
    )
    if not context_ok:
        return None
    for system, patterns in ROTATION_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return system
    return None


def infer_system_from_value(value: object, fallback_crop: str | None) -> str:
    if not pd.isna(value):
        text = str(value)
        rotation = explicit_rotation(text, require_rotation_context=False)
        if rotation:
            return rotation
        hits = crop_hits(text)
        if len(hits) == 1:
            return hits[0]
    return fallback_crop or "Unknown"


def infer_system_from_path(path: Path, fallback_crop: str | None) -> str:
    for segment in reversed(path.parts):
        rotation = explicit_rotation(segment, require_rotation_context=True)
        if rotation:
            return rotation
    return fallback_crop or "Unknown"


def infer_rotation(text: str, fallback_crop: str | None) -> str:
    """Backward-compatible system inference used by tests and external scripts."""
    rotation = explicit_rotation(text, require_rotation_context=False)
    if rotation:
        return rotation
    return fallback_crop or "Unknown"


def infer_regime(text: str) -> str:
    segments = [s for s in re.split(r"[\\/]+", str(text)) if s]
    for segment in reversed(segments):
        low = normalize_name(segment)
        if any(k in low for k in [
            "autoirrigated", "auto_irrigated", "irrigated", "irrigation",
            "water_y", "water_yes",
        ]):
            return "Irrigated"
        if "rainfed" in low or "dryland" in low:
            return "Rainfed"
        if any(k in low for k in ["potential", "non_water_limited", "water_n"]):
            return "Potential"
    return "Unknown"


def normalize_regime_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    direct = str(value).strip().casefold()
    if direct in {"r", "rf", "rainfed", "dryland"}:
        return "Rainfed"
    if direct in {
        "i", "ir", "irr", "irrigated", "auto-irrigated", "auto irrigated",
        "autoirrigated",
    }:
        return "Irrigated"
    if direct in {"p", "pot", "potential", "non-water-limited", "non water limited"}:
        return "Potential"
    result = infer_regime(str(value))
    return result if result != "Unknown" else None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, low_memory=False)
            if len(df.columns) == 1:
                df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
            return df
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def year_validity(series: pd.Series) -> tuple[int, float]:
    values = pd.to_numeric(series, errors="coerce")
    nonmissing = values.notna().sum()
    if nonmissing == 0:
        return 0, 0.0
    valid = values.between(1981, 2018, inclusive="both")
    return int(valid.sum()), float(valid.sum() / nonmissing)


def select_year_column(raw: pd.DataFrame) -> str | None:
    """Find the year by name first, then by actual 1981–2018 values."""
    named = select_column(raw.columns, YEAR_ALIASES)
    if named is not None:
        count, share = year_validity(raw[named])
        if count > 0 and share >= 0.25:
            return named

    candidates: list[tuple[int, float, int, str]] = []
    for position, col in enumerate(raw.columns):
        count, share = year_validity(raw[col])
        if count == 0 or share < 0.80:
            continue
        norm = normalize_name(col)
        # Prefer obvious year/station fields and avoid seven-digit DSSAT dates.
        name_bonus = 2 if any(token in norm for token in ["year", "wsta", "hyear"]) else 0
        candidates.append((name_bonus, share, count, -position, col))
    if not candidates:
        return None
    return max(candidates)[-1]


def is_shifted_dssat_export(raw: pd.DataFrame) -> bool:
    """Detect exports whose headers are displaced after O# / P#.

    In the supplied continuous-crop files, P# contains crop codes, CR contains
    model codes, and WSTA contains the calendar year. This detector uses those
    values rather than relying on the misleading labels.
    """
    normalized = {normalize_name(c): c for c in raw.columns}
    needed = {"p", "cr", "wsta", "cwam", "hwam"}
    if not needed.issubset(normalized):
        return False

    p_col = normalized["p"]
    cr_col = normalized["cr"]
    wsta_col = normalized["wsta"]
    sample = raw.head(5000)
    crop_share = sample[p_col].map(normalize_crop_value).notna().mean()
    model_share = sample[cr_col].astype(str).str.contains(
        r"(?:CER|GRO|CSM)\d*", case=False, regex=True, na=False
    ).mean()
    _, year_share = year_validity(sample[wsta_col])
    return crop_share >= 0.50 and model_share >= 0.50 and year_share >= 0.80


def detect_regime_column(raw: pd.DataFrame) -> str | None:
    explicit = select_column(raw.columns, REGIME_ALIASES)
    if explicit is not None:
        return explicit
    # Several continuous outputs use a generic column named ``system`` for
    # values such as rainfed and autoirrigated.
    system_named = {normalize_name(c): c for c in raw.columns}.get("system")
    if system_named is not None:
        recognized = raw[system_named].head(5000).map(normalize_regime_value).notna().mean()
        if recognized >= 0.50:
            return system_named
    return None


def identify_wide_yield_columns(columns: Iterable[str]) -> list[tuple[str, str, str | None]]:
    """Identify crop-specific wide columns such as SB_rainfed_yield_kg_ha."""
    found: list[tuple[str, str, str | None]] = []
    for col in columns:
        norm = normalize_name(col)
        if not any(token in norm for token in ["yield", "hwam", "hwah"]):
            continue
        crop = infer_crop(norm)
        if crop is None:
            continue
        regime = normalize_regime_value(norm)
        found.append((col, crop, regime))
    return found


def build_normalized_frame(
    *,
    raw: pd.DataFrame,
    path: Path,
    root: Path,
    year_col: str,
    yield_col: str,
    crop_col: str | None,
    fixed_crop: str | None,
    fixed_regime: str | None,
    irrigation_col: str | None,
    rainfall_col: str | None,
    site_col: str | None,
    latitude_col: str | None,
    longitude_col: str | None,
    system_col: str | None,
    regime_col: str | None,
    shifted: bool,
) -> pd.DataFrame:
    path_text = str(path)
    inferred_crop = fixed_crop or infer_crop(path_text)
    inferred_system = infer_system_from_path(path, inferred_crop)
    inferred_regime = fixed_regime or infer_regime(path_text)

    out = pd.DataFrame(index=raw.index)
    out["year"] = pd.to_numeric(raw[year_col], errors="coerce")
    out["yield_kg_ha"] = pd.to_numeric(raw[yield_col], errors="coerce")

    if fixed_crop is not None:
        out["crop_code"] = fixed_crop
    elif crop_col is not None:
        out["crop_code"] = raw[crop_col].map(normalize_crop_value)
        if inferred_crop is not None:
            out["crop_code"] = out["crop_code"].fillna(inferred_crop)
    elif inferred_crop is not None:
        out["crop_code"] = inferred_crop
    else:
        out["crop_code"] = np.nan

    if regime_col is not None:
        out["water_regime"] = raw[regime_col].map(normalize_regime_value)
        if inferred_regime != "Unknown":
            out["water_regime"] = out["water_regime"].fillna(inferred_regime)
    else:
        out["water_regime"] = inferred_regime
    if fixed_regime is not None:
        out["water_regime"] = fixed_regime

    if irrigation_col is not None:
        out["irrigation_mm"] = pd.to_numeric(raw[irrigation_col], errors="coerce")
    else:
        out["irrigation_mm"] = np.nan
    # Rainfed simulations have no applied irrigation. This also protects
    # against shifted headers where IRCM actually contains rainfall.
    out.loc[out["water_regime"].eq("Rainfed"), "irrigation_mm"] = 0.0

    if rainfall_col is not None:
        out["rainfall_mm"] = pd.to_numeric(raw[rainfall_col], errors="coerce")
    else:
        out["rainfall_mm"] = np.nan

    if system_col is not None:
        out["base_system"] = [
            infer_system_from_value(value, crop)
            for value, crop in zip(raw[system_col], out["crop_code"])
        ]
        unresolved = out["base_system"].eq("Unknown")
        out.loc[unresolved, "base_system"] = inferred_system
    else:
        out["base_system"] = inferred_system

    if inferred_system in {"SB-MZ", "SB-MZ-SG", "SB-MZ-SG-WT"}:
        out["base_system"] = inferred_system

    out["rotation"] = out["base_system"]
    out["cropping_system"] = out["base_system"] + " | " + out["water_regime"]
    out["site_id"] = raw[site_col].astype(str) if site_col else path.parent.name
    if latitude_col is not None:
        out["latitude"] = pd.to_numeric(raw[latitude_col], errors="coerce")
    else:
        out["latitude"] = np.nan
    if longitude_col is not None:
        out["longitude"] = pd.to_numeric(raw[longitude_col], errors="coerce")
    else:
        out["longitude"] = np.nan

    missing_coordinates = out["latitude"].isna() | out["longitude"].isna()
    if missing_coordinates.any():
        parsed = out.loc[missing_coordinates, "site_id"].map(parse_site_coordinates)
        out.loc[missing_coordinates, "latitude"] = [pair[0] for pair in parsed]
        out.loc[missing_coordinates, "longitude"] = [pair[1] for pair in parsed]

    out["source_file"] = str(path.relative_to(root))
    out["source_schema"] = "shifted_dssat" if shifted else "standard"
    return out


def standardize_file(path: Path, root: Path) -> tuple[pd.DataFrame | None, dict]:
    report = {
        "file": str(path.relative_to(root)),
        "status": "SKIPPED",
        "reason": "",
        "rows_read": 0,
        "rows_kept": 0,
        "year_column": "",
        "yield_column": "",
        "irrigation_column": "",
        "rainfall_column": "",
        "crop_column": "",
        "system_column": "",
        "regime_column": "",
        "schema": "",
        "inferred_crop": "",
        "inferred_system": "",
        "inferred_regime": "",
    }
    try:
        raw = read_table(path)
    except Exception as exc:
        report["status"] = "ERROR"
        report["reason"] = f"read_error: {exc}"
        return None, report

    report["rows_read"] = len(raw)
    if raw.empty:
        report["reason"] = "empty_file"
        return None, report

    path_text = str(path)
    inferred_crop = infer_crop(path_text)
    inferred_system = infer_system_from_path(path, inferred_crop)
    inferred_regime = infer_regime(path_text)
    shifted = is_shifted_dssat_export(raw)
    report.update(
        schema="shifted_dssat" if shifted else "standard",
        inferred_crop=inferred_crop or "",
        inferred_system=inferred_system,
        inferred_regime=inferred_regime,
    )

    site_col = select_column(raw.columns, SITE_ALIASES)
    latitude_col = select_column(raw.columns, LATITUDE_ALIASES)
    longitude_col = select_column(raw.columns, LONGITUDE_ALIASES)
    regime_col = detect_regime_column(raw)
    system_col = select_column(raw.columns, SYSTEM_ALIASES)

    frames: list[pd.DataFrame] = []

    if shifted:
        normalized = {normalize_name(c): c for c in raw.columns}
        year_col = normalized["wsta"]
        crop_col = normalized["p"]
        # CWAM contains the actual HWAM in these one-position-shifted files.
        yield_col = normalized["cwam"]
        irrigation_col = normalized.get("ircm")
        # With FPWAM present, the positional shift continues into the water
        # fields; otherwise it has already realigned by IRCM.
        if "fpwam" in normalized:
            rainfall_col = normalized.get("ircm")
        else:
            rainfall_col = normalized.get("prcm")
        system_col = None  # the generic system column is the water regime here

        frame = build_normalized_frame(
            raw=raw,
            path=path,
            root=root,
            year_col=year_col,
            yield_col=yield_col,
            crop_col=crop_col,
            fixed_crop=None,
            fixed_regime=None,
            irrigation_col=irrigation_col,
            rainfall_col=rainfall_col,
            site_col=site_col,
            latitude_col=latitude_col,
            longitude_col=longitude_col,
            system_col=system_col,
            regime_col=regime_col,
            shifted=True,
        )
        frames.append(frame)
        report.update(
            year_column=year_col,
            yield_column=yield_col,
            irrigation_column=irrigation_col or "",
            rainfall_column=rainfall_col or "",
            crop_column=crop_col,
            regime_column=regime_col or "",
        )
    else:
        year_col = select_year_column(raw)
        yield_col = select_column(raw.columns, YIELD_ALIASES)
        irrigation_col = select_column(raw.columns, IRRIGATION_ALIASES)
        rainfall_col = select_column(raw.columns, RAINFALL_ALIASES)
        crop_col = select_column(raw.columns, CROP_ALIASES)
        wide_yields = identify_wide_yield_columns(raw.columns) if yield_col is None else []

        if year_col is None or (yield_col is None and not wide_yields):
            report["reason"] = "missing_year_or_yield_column"
            return None, report

        if yield_col is not None:
            frame = build_normalized_frame(
                raw=raw,
                path=path,
                root=root,
                year_col=year_col,
                yield_col=yield_col,
                crop_col=crop_col,
                fixed_crop=None,
                fixed_regime=None,
                irrigation_col=irrigation_col,
                rainfall_col=rainfall_col,
                site_col=site_col,
                latitude_col=latitude_col,
                longitude_col=longitude_col,
                system_col=system_col,
                regime_col=regime_col,
                shifted=False,
            )
            frames.append(frame)
            report.update(
                year_column=year_col,
                yield_column=yield_col,
                irrigation_column=irrigation_col or "",
                rainfall_column=rainfall_col or "",
                crop_column=crop_col or "",
                system_column=system_col or "",
                regime_column=regime_col or "",
            )
        else:
            for wide_col, crop, regime_from_col in wide_yields:
                frame = build_normalized_frame(
                    raw=raw,
                    path=path,
                    root=root,
                    year_col=year_col,
                    yield_col=wide_col,
                    crop_col=None,
                    fixed_crop=crop,
                    fixed_regime=regime_from_col,
                    irrigation_col=irrigation_col,
                    rainfall_col=rainfall_col,
                    site_col=site_col,
                    latitude_col=latitude_col,
                    longitude_col=longitude_col,
                    system_col=system_col,
                    regime_col=regime_col,
                    shifted=False,
                )
                frames.append(frame)
            report.update(
                year_column=year_col,
                yield_column=";".join(col for col, _, _ in wide_yields),
                irrigation_column=irrigation_col or "",
                rainfall_column=rainfall_col or "",
                crop_column="wide_column_names",
                system_column=system_col or "",
                regime_column=regime_col or "",
            )

    out = pd.concat(frames, ignore_index=True)

    if out["crop_code"].notna().sum() == 0:
        report["reason"] = "unrecognized_crop_values_and_no_path_fallback"
        return None, report

    out = out[
        out["year"].between(1981, 2018, inclusive="both")
        & out["yield_kg_ha"].notna()
        & out["crop_code"].isin(VALID_CROPS)
        & out["water_regime"].isin(EXPECTED_REGIMES)
        & out["base_system"].isin(EXPECTED_SYSTEMS)
    ].copy()
    out["year"] = out["year"].astype(int)
    out["yield_kg_ha"] = pd.to_numeric(out["yield_kg_ha"], errors="coerce")
    out["irrigation_mm"] = pd.to_numeric(out["irrigation_mm"], errors="coerce").clip(lower=0)
    out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce").clip(lower=0)

    if out.empty:
        report["reason"] = "no_valid_1981_2018_rows_after_standardization"
        return None, report

    report["status"] = "USED"
    report["rows_kept"] = len(out)
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable state-level cropping-systems dataset for Streamlit."
    )
    parser.add_argument(
        "--root",
        default=r"H:\All cropping systems result",
        help="Root folder containing all DSSAT/processed simulation results.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cropping_systems_long.csv",
        help="Output CSV path, relative to the repository unless absolute.",
    )
    parser.add_argument(
        "--report",
        default="data/processed/preparation_report.csv",
        help="Data preparation diagnostic report.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    output = Path(args.output)
    report_path = Path(args.report)
    if not output.is_absolute():
        output = repo_root / output
    if not report_path.is_absolute():
        report_path = repo_root / report_path

    if not root.exists():
        raise SystemExit(
            f"Input root not found: {root}\n"
            "Run this script on the Windows computer that contains the H: drive, "
            "or provide --root with the correct folder."
        )

    candidates = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and not p.name.startswith("~$")
    )
    frames: list[pd.DataFrame] = []
    reports: list[dict] = []

    for index, file_path in enumerate(candidates, start=1):
        try:
            frame, report = standardize_file(file_path, root)
        except Exception as exc:
            frame = None
            report = {
                "file": str(file_path.relative_to(root)),
                "status": "ERROR",
                "reason": f"unexpected_standardization_error: {type(exc).__name__}: {exc}",
                "rows_read": 0,
                "rows_kept": 0,
                "year_column": "",
                "yield_column": "",
                "irrigation_column": "",
                "rainfall_column": "",
                "crop_column": "",
                "system_column": "",
                "regime_column": "",
                "schema": "",
                "inferred_crop": "",
                "inferred_system": "",
                "inferred_regime": "",
            }
        reports.append(report)
        if frame is not None:
            frames.append(frame)
        if index % 25 == 0 or index == len(candidates):
            used = sum(1 for item in reports if item["status"] == "USED")
            print(f"checked {index:,}/{len(candidates):,} files | used={used:,}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(reports)
    report_df.to_csv(report_path, index=False)

    if not frames:
        raise SystemExit(
            "No usable simulation data were found. Review "
            f"{report_path} for missing or unrecognized columns."
        )

    all_rows = pd.concat(frames, ignore_index=True)
    # Apply explicit system-specific corrections before either the state-level
    # or spatial products are aggregated. This preserves all yield values while
    # preventing the SB-MZ-SG rainfed source from being published as Irrigated.
    all_rows = apply_regime_relabels(all_rows)

    # Preserve a compressed site-year dataset for interactive spatial maps.
    # Invalid coordinates are excluded; missing values within an individual
    # map are filled on demand from the complete master grid in the app.
    all_rows["latitude"] = pd.to_numeric(all_rows.get("latitude"), errors="coerce")
    all_rows["longitude"] = pd.to_numeric(all_rows.get("longitude"), errors="coerce")
    spatial_source = all_rows[
        all_rows["latitude"].between(36.0, 41.5, inclusive="both")
        & all_rows["longitude"].between(-103.5, -93.0, inclusive="both")
    ].copy()
    spatial_group_cols = [
        "year", "cropping_system", "base_system", "rotation",
        "crop_code", "water_regime", "site_id", "latitude", "longitude",
    ]
    spatial = (
        spatial_source.groupby(spatial_group_cols, as_index=False, dropna=False)
        .agg(
            yield_kg_ha=("yield_kg_ha", "mean"),
            irrigation_mm=("irrigation_mm", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            source_files=("source_file", "nunique"),
        )
        .sort_values(["base_system", "water_regime", "year", "site_id", "crop_code"])
    )

    group_cols = [
        "year", "cropping_system", "base_system", "rotation",
        "crop_code", "water_regime",
    ]
    aggregated = (
        all_rows.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            yield_kg_ha=("yield_kg_ha", "mean"),
            irrigation_mm=("irrigation_mm", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            n_sites=("site_id", "nunique"),
            source_files=("source_file", "nunique"),
        )
        .sort_values(["base_system", "water_regime", "year", "crop_code"])
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output, index=False)
    spatial_output = output.parent / "cropping_systems_spatial.parquet"
    master_grid_output = output.parent / "master_site_grid.csv"
    if spatial.empty:
        print("WARNING: No valid site coordinates were found; spatial maps will be unavailable.")
        master_grid = pd.DataFrame(
            columns=["site_id", "latitude", "longitude", "grid_site_status"]
        )
    else:
        spatial.to_parquet(spatial_output, index=False, compression="snappy")
        master_grid = build_authoritative_master_grid(
            spatial[["site_id", "latitude", "longitude"]]
        )
        master_grid.to_csv(master_grid_output, index=False)
        restored_count = 0
        print(
            f"Authoritative master grid: {len(master_grid):,} original simulation sites"
        )

    found_systems = [s for s in EXPECTED_SYSTEMS if s in set(aggregated["base_system"])]
    missing_systems = [s for s in EXPECTED_SYSTEMS if s not in set(aggregated["base_system"])]
    coverage = (
        aggregated.groupby(["base_system", "water_regime"], as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            years=("year", "nunique"),
            crops=("crop_code", lambda x: ",".join(sorted(set(x)))),
            rows=("year", "size"),
            mean_sites=("n_sites", "mean"),
        )
    )
    coverage.to_csv(output.parent / "system_coverage.csv", index=False)

    summary = {
        "input_root": str(root),
        "candidate_files": len(candidates),
        "used_files": int((report_df["status"] == "USED").sum()),
        "standardized_rows": len(all_rows),
        "output_rows": len(aggregated),
        "years": [int(aggregated["year"].min()), int(aggregated["year"].max())],
        "found_systems": found_systems,
        "missing_expected_systems": missing_systems,
        "water_regimes": [r for r in EXPECTED_REGIMES if r in set(aggregated["water_regime"])],
        "shifted_dssat_files_used": int(
            ((report_df["status"] == "USED") & (report_df["schema"] == "shifted_dssat")).sum()
        ),
        "spatial_rows": int(len(spatial)),
        "spatial_sites": int(spatial["site_id"].nunique()) if not spatial.empty else 0,
        "master_grid_sites": int(len(master_grid)),
        "restored_master_grid_sites": 0,
        "spatial_output": str(spatial_output),
        "master_grid_output": str(master_grid_output),
        "output": str(output),
        "report": str(report_path),
        "coverage": str(output.parent / "system_coverage.csv"),
    }
    (output.parent / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
