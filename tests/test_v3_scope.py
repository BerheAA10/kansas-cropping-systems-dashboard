import pandas as pd
import pytest

from src.data_loader import apply_decision_scope


def test_v3_excludes_continuous_wheat_but_retains_rotation_wheat():
    data = pd.DataFrame(
        {
            "base_system": ["WT", "Continuous wheat", "SB-MZ-SG-WT", "MZ"],
            "crop_code": ["WT", "WT", "WT", "MZ"],
            "yield_kg_ha": [7000, 7100, 6500, 9000],
        }
    )

    scoped = apply_decision_scope(data)

    assert set(scoped["base_system"]) == {"SB-MZ-SG-WT", "MZ"}
    assert ((scoped["base_system"] == "SB-MZ-SG-WT") & (scoped["crop_code"] == "WT")).any()


def test_v3_scope_requires_base_system():
    with pytest.raises(ValueError, match="base_system"):
        apply_decision_scope(pd.DataFrame({"crop_code": ["WT"]}))
