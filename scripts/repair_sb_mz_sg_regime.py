from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

TARGET_SYSTEM = "SB-MZ-SG"
SOURCE_REGIME = "Irrigated"
TARGET_REGIME = "Rainfed"
EXPECTED_REGIMES = {"Rainfed", "Potential"}
EXPECTED_CROPS = {"MZ", "SB", "SG"}
EXPECTED_YEARS = set(range(1981, 2019))


def banner(text: str) -> None:
    print("=" * 118)
    print(text)
    print("=" * 118)


def summarize_target(df: pd.DataFrame, label: str) -> None:
    subset = df[df["base_system"].astype(str).str.strip().eq(TARGET_SYSTEM)].copy()
    print(f"{label}: {len(subset):,} {TARGET_SYSTEM} rows")
    if subset.empty:
        print("  No target rows found.")
        return
    stats = (
        subset.groupby("water_regime", dropna=False)
        .agg(
            rows=("year", "size"),
            years=("year", "nunique"),
            mean_yield_kg_ha=("yield_kg_ha", "mean"),
            mean_irrigation_mm=("irrigation_mm", "mean"),
        )
        .reset_index()
        .sort_values("water_regime")
    )
    for row in stats.itertuples(index=False):
        print(
            f"  {row.water_regime}: rows={int(row.rows):,}, years={int(row.years)}, "
            f"mean_yield={float(row.mean_yield_kg_ha):,.1f} kg/ha, "
            f"mean_irrigation={float(row.mean_irrigation_mm):,.1f} mm"
        )


def correct_frame(df: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, int]:
    required = {
        "year",
        "cropping_system",
        "base_system",
        "rotation",
        "crop_code",
        "water_regime",
        "yield_kg_ha",
        "irrigation_mm",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{label} is missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["base_system"] = out["base_system"].astype(str).str.strip()
    out["water_regime"] = out["water_regime"].astype(str).str.strip()
    target = out["base_system"].eq(TARGET_SYSTEM)
    source = target & out["water_regime"].str.casefold().eq(SOURCE_REGIME.casefold())
    existing_target = target & out["water_regime"].str.casefold().eq(TARGET_REGIME.casefold())

    if source.any() and existing_target.any():
        raise RuntimeError(
            f"{label} contains both {TARGET_SYSTEM} {SOURCE_REGIME} and {TARGET_REGIME} rows. "
            "The hotfix stopped to avoid combining two possibly distinct datasets."
        )

    changed = int(source.sum())
    if changed:
        # This is a metadata correction only. Preserve every yield, rainfall,
        # crop, year, site, and source-count value exactly.
        out.loc[source, "water_regime"] = TARGET_REGIME
        out.loc[source, "irrigation_mm"] = 0.0
        out.loc[source, "cropping_system"] = f"{TARGET_SYSTEM} | {TARGET_REGIME}"

    # Rebuild all labels consistently after the correction.
    out["cropping_system"] = (
        out["base_system"].astype(str).str.strip()
        + " | "
        + out["water_regime"].astype(str).str.strip()
    )

    subset = out[out["base_system"].eq(TARGET_SYSTEM)].copy()
    regimes = set(subset["water_regime"].dropna().astype(str))
    if SOURCE_REGIME in regimes:
        raise RuntimeError(f"{label}: {SOURCE_REGIME} rows remain for {TARGET_SYSTEM}.")
    missing_regimes = EXPECTED_REGIMES.difference(regimes)
    if missing_regimes:
        raise RuntimeError(
            f"{label}: expected {TARGET_SYSTEM} regimes are missing: "
            + ", ".join(sorted(missing_regimes))
        )
    unexpected = regimes.difference(EXPECTED_REGIMES)
    if unexpected:
        raise RuntimeError(
            f"{label}: unexpected {TARGET_SYSTEM} regimes remain: "
            + ", ".join(sorted(unexpected))
        )

    for regime in sorted(EXPECTED_REGIMES):
        regime_rows = subset[subset["water_regime"].eq(regime)]
        years = set(pd.to_numeric(regime_rows["year"], errors="coerce").dropna().astype(int))
        crops = set(regime_rows["crop_code"].dropna().astype(str).str.upper())
        if years != EXPECTED_YEARS:
            missing_years = sorted(EXPECTED_YEARS.difference(years))
            extra_years = sorted(years.difference(EXPECTED_YEARS))
            raise RuntimeError(
                f"{label}: {TARGET_SYSTEM} {regime} does not have the exact 1981-2018 coverage. "
                f"missing={missing_years}, extra={extra_years}"
            )
        if crops != EXPECTED_CROPS:
            raise RuntimeError(
                f"{label}: {TARGET_SYSTEM} {regime} crop set is {sorted(crops)}, "
                f"expected {sorted(EXPECTED_CROPS)}."
            )

    rainfed_irrigation = pd.to_numeric(
        subset.loc[subset["water_regime"].eq(TARGET_REGIME), "irrigation_mm"],
        errors="coerce",
    ).fillna(0.0)
    if not rainfed_irrigation.eq(0.0).all():
        raise RuntimeError(f"{label}: relabeled Rainfed rows still contain applied irrigation.")

    return out, changed


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(path)


def rebuild_coverage(summary: pd.DataFrame, output: Path) -> None:
    if not output.exists():
        return
    coverage = (
        summary.groupby(["base_system", "water_regime"], as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            years=("year", "nunique"),
            crops=("crop_code", lambda x: ",".join(sorted(set(x.astype(str))))),
            rows=("year", "size"),
            mean_sites=("n_sites", "mean"),
        )
    )
    atomic_write_csv(coverage, output)


def patch_dataset_root(root: Path, backup_root: Path, label: str) -> dict[str, object]:
    processed = root / "data" / "processed"
    summary_path = processed / "cropping_systems_long.csv"
    spatial_path = processed / "cropping_systems_spatial.parquet"
    coverage_path = processed / "system_coverage.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {label} summary CSV: {summary_path}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"Missing {label} spatial Parquet: {spatial_path}")

    backup_dir = backup_root / label.lower().replace(" ", "_")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for source in [summary_path, spatial_path, coverage_path]:
        if source.exists():
            shutil.copy2(source, backup_dir / source.name)

    summary = pd.read_csv(summary_path, low_memory=False)
    spatial = pd.read_parquet(spatial_path)

    banner(f"{label}: BEFORE")
    summarize_target(summary, "Summary")
    summarize_target(spatial, "Spatial")

    corrected_summary, changed_summary = correct_frame(summary, label=f"{label} summary")
    corrected_spatial, changed_spatial = correct_frame(spatial, label=f"{label} spatial")

    atomic_write_csv(corrected_summary, summary_path)
    atomic_write_parquet(corrected_spatial, spatial_path)
    rebuild_coverage(corrected_summary, coverage_path)

    banner(f"{label}: AFTER")
    summarize_target(corrected_summary, "Summary")
    summarize_target(corrected_spatial, "Spatial")
    print(f"  Relabeled summary rows: {changed_summary:,}")
    print(f"  Relabeled spatial rows: {changed_spatial:,}")

    return {
        "summary_rows_relabelled": changed_summary,
        "spatial_rows_relabelled": changed_spatial,
        "summary_path": str(summary_path),
        "spatial_path": str(spatial_path),
    }


def copy_corrected_data(project: Path, repo: Path) -> None:
    source = project / "data" / "processed"
    destination = repo / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "cropping_systems_long.csv",
        "cropping_systems_spatial.parquet",
        "system_coverage.csv",
    ]:
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, destination / name)


def validate_repo(repo: Path) -> None:
    summary = pd.read_csv(repo / "data" / "processed" / "cropping_systems_long.csv")
    spatial = pd.read_parquet(repo / "data" / "processed" / "cropping_systems_spatial.parquet")
    correct_frame(summary, label="GitHub summary validation")
    correct_frame(spatial, label="GitHub spatial validation")


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Relabel the current SB-MZ-SG Irrigated records as Rainfed without changing yields."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=home / "Downloads" / "kansas_cropping_systems_streamlit" / "kansas_cropping_systems_streamlit",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=home / "Downloads" / "kansas-cropping-systems-dashboard-github",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    repo = args.repo.resolve()
    if not project.exists():
        raise FileNotFoundError(f"Local project not found: {project}")
    if not repo.exists():
        raise FileNotFoundError(f"GitHub working repository not found: {repo}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project / "backups" / f"sb_mz_sg_regime_hotfix_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    result = patch_dataset_root(project, backup_root, "Local project")
    copy_corrected_data(project, repo)
    validate_repo(repo)

    report = {
        "status": "PASS",
        "target_system": TARGET_SYSTEM,
        "old_regime": SOURCE_REGIME,
        "new_regime": TARGET_REGIME,
        "yield_values_changed": False,
        "backup": str(backup_root),
        **result,
    }
    report_path = backup_root / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    banner("SB-MZ-SG REGIME CORRECTION: PASS")
    print("SB-MZ-SG now has Rainfed and Potential only.")
    print("No yield values were changed.")
    print("Rainfed irrigation was set to 0 mm.")
    print(f"Backup: {backup_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
