from scripts.prepare_simulation_data import infer_crop, infer_rotation


def test_continuous_crop_inference_ignores_method_reference():
    path = r"H:\All cropping systems result\Continuous_Wheat\AutoIrrigated_Exact_Maize_Method\wheat_statewide.csv"
    crop = infer_crop(path)
    assert crop == "WT"
    assert infer_rotation(path, crop) == "WT"


def test_all_continuous_system_codes_are_supported():
    cases = {
        r"H:\results\Continuous_Maize\master.csv": "MZ",
        r"H:\results\Continuous_Sorghum\master.csv": "SG",
        r"H:\results\Continuous_Wheat\master.csv": "WT",
        r"H:\results\Continuous_Soybean\master.csv": "SB",
    }
    for path, expected in cases.items():
        crop = infer_crop(path)
        assert crop == expected
        assert infer_rotation(path, crop) == expected


def test_rotation_inference_is_preserved():
    path = r"H:\results\SB-MZ-SG-WT\Rainfed\master.csv"
    crop = infer_crop(path)
    assert infer_rotation(path, crop) == "SB-MZ-SG-WT"


def test_rotation_year_alias_and_hwam_are_supported(tmp_path):
    import pandas as pd
    from scripts.prepare_simulation_data import standardize_file

    root = tmp_path
    path = root / "SB-MZ-SG-WT" / "statewide_rotation_summary_1981_2018_rainfed.csv"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "site": ["37_0417N_094_6250W"],
            "crop_code": ["SB"],
            "rotation_year": [1981],
            "hwam": [1027],
            "irrigation_mm": [0],
        }
    ).to_csv(path, index=False)

    frame, report = standardize_file(path, root)
    assert report["status"] == "USED"
    assert frame is not None
    assert frame.loc[0, "year"] == 1981
    assert frame.loc[0, "yield_kg_ha"] == 1027
    assert frame.loc[0, "base_system"] == "SB-MZ-SG-WT"


def test_shifted_continuous_dssat_export_is_repaired(tmp_path):
    import pandas as pd
    from scripts.prepare_simulation_data import standardize_file

    root = tmp_path
    path = root / "SG" / "sorghum_autoirrigated_statewide_summary_1981_2018-1.csv"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "system": ["autoirrigated"],
            "site_name": ["37_0417N_094_6250W"],
            "P#": ["SG"],
            "CR": ["SGCER048"],
            "WSTA": [1981],
            "CWAM": [8075],
            "HWAM": [8075],
            "IRCM": [125],
            "PRCM": [734],
        }
    ).to_csv(path, index=False)

    frame, report = standardize_file(path, root)
    assert report["status"] == "USED"
    assert report["schema"] == "shifted_dssat"
    assert frame is not None
    assert frame.loc[0, "crop_code"] == "SG"
    assert frame.loc[0, "year"] == 1981
    assert frame.loc[0, "yield_kg_ha"] == 8075
    assert frame.loc[0, "irrigation_mm"] == 125
    assert frame.loc[0, "water_regime"] == "Irrigated"
