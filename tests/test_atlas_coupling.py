from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from remind_mfa.atlas_coupling import (
    AtlasCouplingError,
    AtlasScenarioPkBudg,
    _adjust_plastics_trade,
    _ras_balance,
    aggregate_bilateral_trade,
    copy_demand_to_atlas,
    copy_trade_to_mfa,
    get_model_spec,
    load_atlas_trade_projection,
    scenario_pkbudg_is_effective,
)
from remind_mfa.common.helpers import ModelNames


def test_copy_steel_demand_converts_schema(tmp_path):
    source = tmp_path / "steel_demand.csv"
    target = tmp_path / "REMIND_MFA" / "ip_market__fabrication.csv"
    pd.DataFrame(
        {
            "Time": [2026, 2025],
            "Region": ["B", "A"],
            "steel_demand": [2.5, 1.0],
        }
    ).to_csv(source, index=False)

    result = copy_demand_to_atlas(ModelNames.STEEL, source, target)

    expected = pd.DataFrame({"Time": [2026, 2025], "Region": ["B", "A"], "value": [2.5, 1.0]})
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(pd.read_csv(target), expected)


def test_aggregate_bilateral_trade_excludes_domestic_flows():
    bilateral = pd.DataFrame(
        {
            "i": ["A", "B", "A"],
            "j": ["B", "A", "A"],
            "year": [2025, 2025, 2025],
            "quantity": [2.0, 3.0, 99.0],
        }
    )

    imports, exports = aggregate_bilateral_trade(bilateral)

    assert imports.to_dict("records") == [
        {"year": 2025, "region": "A", "quantity": 3.0},
        {"year": 2025, "region": "B", "quantity": 2.0},
    ]
    assert exports.to_dict("records") == [
        {"year": 2025, "region": "A", "quantity": 2.0},
        {"year": 2025, "region": "B", "quantity": 3.0},
    ]


def test_aggregate_bilateral_trade_sums_flows():
    bilateral = pd.DataFrame(
        {
            "i": ["A", "A", "A"],
            "j": ["B", "B", "A"],
            "year": [2025, 2025, 2025],
            "quantity": [2.0, 3.0, 99.0],
        }
    )

    imports, exports = aggregate_bilateral_trade(bilateral)

    assert imports.to_dict("records") == [{"year": 2025, "region": "B", "quantity": 5.0}]
    assert exports.to_dict("records") == [{"year": 2025, "region": "A", "quantity": 5.0}]


def _write_historic_plastics_trade(tmp_path: Path, monkeypatch, imports: str, exports: str) -> None:
    parameters_path = tmp_path / "data_in" / "parameters"
    parameters_path.mkdir(parents=True)
    header = "* note: dimensions: (Historic Time,Region,Type,Material,value)\n"
    for flow, rows in (("imports", imports), ("exports", exports)):
        (parameters_path / f"pl_primary_his_{flow}.cs4r").write_text(
            header + rows, encoding="utf-8"
        )
    monkeypatch.setattr("remind_mfa.atlas_coupling.PROJECT_ROOT", tmp_path)


def test_adjust_plastics_trade_keeps_latest_historic_shares_if_balanced(tmp_path, monkeypatch):
    _write_historic_plastics_trade(
        tmp_path,
        monkeypatch,
        imports="2023,EUR,Plastics,LDPE,100\n"
        "2024,EUR,Plastics,HDPE,3\n"
        "2024,EUR,Plastics,LDPE,1\n"
        "2024,CAN,Plastics,LDPE,1\n",
        exports="2024,EUR,Plastics,HDPE,1\n2024,EUR,Plastics,LDPE,1\n2024,CAN,Plastics,LDPE,1\n",
    )

    imports, exports = _adjust_plastics_trade(
        pd.DataFrame({"year": [2025, 2025], "region": ["EUR", "CAN"], "quantity": [40.0, 60.0]}),
        pd.DataFrame({"year": [2025, 2025], "region": ["EUR", "CAN"], "quantity": [60.0, 40.0]}),
    )

    assert imports.to_dict("records") == [
        {"year": 2025, "region": "CAN", "type": "Plastics", "material": "HDPE", "quantity": 0.0},
        {"year": 2025, "region": "CAN", "type": "Plastics", "material": "LDPE", "quantity": 60.0},
        {"year": 2025, "region": "EUR", "type": "Plastics", "material": "HDPE", "quantity": 30.0},
        {"year": 2025, "region": "EUR", "type": "Plastics", "material": "LDPE", "quantity": 10.0},
    ]
    assert exports.to_dict("records") == [
        {"year": 2025, "region": "CAN", "type": "Plastics", "material": "HDPE", "quantity": 0.0},
        {"year": 2025, "region": "CAN", "type": "Plastics", "material": "LDPE", "quantity": 40.0},
        {"year": 2025, "region": "EUR", "type": "Plastics", "material": "HDPE", "quantity": 30.0},
        {"year": 2025, "region": "EUR", "type": "Plastics", "material": "LDPE", "quantity": 30.0},
    ]


def test_adjust_plastics_trade_balances_polymers_and_keeps_region_totals(tmp_path, monkeypatch):
    # Historic shares imply HDPE:LDPE odds of 1 for imports and 3 for exports. RAS settles on
    # their geometric mean sqrt(3) for both flows.
    _write_historic_plastics_trade(
        tmp_path,
        monkeypatch,
        imports="2024,A,Plastics,HDPE,1\n2024,A,Plastics,LDPE,1\n"
        "2024,B,Plastics,HDPE,2\n2024,B,Plastics,LDPE,2\n",
        exports="2024,A,Plastics,HDPE,3\n2024,A,Plastics,LDPE,1\n"
        "2024,B,Plastics,HDPE,6\n2024,B,Plastics,LDPE,2\n",
    )

    imports, exports = _adjust_plastics_trade(
        pd.DataFrame({"year": [2025, 2025], "region": ["A", "B"], "quantity": [1.0, 3.0]}),
        pd.DataFrame({"year": [2025, 2025], "region": ["A", "B"], "quantity": [2.5, 1.5]}),
    )

    for flow, totals in ((imports, [1.0, 3.0]), (exports, [2.5, 1.5])):
        assert flow.groupby("region")["quantity"].sum().tolist() == pytest.approx(totals)
        hdpe_share = flow.loc[flow["material"] == "HDPE", "quantity"].sum() / 4.0
        assert hdpe_share == pytest.approx(3**0.5 / (1 + 3**0.5))
    assert imports.groupby("material")["quantity"].sum().tolist() == pytest.approx(
        exports.groupby("material")["quantity"].sum().tolist()
    )


def test_ras_balance_clears_global_trade_per_good():
    rng = np.random.default_rng(0)
    imports = rng.uniform(size=(3, 4, 5))
    exports = rng.uniform(size=(3, 4, 5))
    exports *= imports.sum(axis=(1, 2), keepdims=True) / exports.sum(axis=(1, 2), keepdims=True)

    balanced_imports, balanced_exports = _ras_balance(imports, exports)

    np.testing.assert_allclose(balanced_imports.sum(axis=2), imports.sum(axis=2))
    np.testing.assert_allclose(balanced_exports.sum(axis=2), exports.sum(axis=2))
    np.testing.assert_allclose(balanced_imports.sum(axis=1), balanced_exports.sum(axis=1))


def test_ras_balance_rejects_infeasible_zero_pattern():
    # A only imports HDPE and B only exports LDPE, so polymer trade can never clear.
    imports = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    exports = np.array([[[0.0, 0.0], [0.0, 1.0]]])

    with pytest.raises(AtlasCouplingError, match="did not converge"):
        _ras_balance(imports, exports)


def test_adjust_plastics_trade_rejects_regions_without_historic_shares(tmp_path, monkeypatch):
    _write_historic_plastics_trade(
        tmp_path,
        monkeypatch,
        imports="2024,A,Plastics,LDPE,1\n",
        exports="2024,A,Plastics,LDPE,1\n2024,B,Plastics,LDPE,1\n",
    )

    with pytest.raises(AtlasCouplingError, match=r"no polymer shares .*\['B'\]"):
        _adjust_plastics_trade(
            pd.DataFrame({"year": [2025, 2025], "region": ["A", "B"], "quantity": [1.0, 1.0]}),
            pd.DataFrame({"year": [2025, 2025], "region": ["A", "B"], "quantity": [1.0, 1.0]}),
        )


def _write_dimensions(tmp_path: Path) -> tuple[Path, Path]:
    regions = tmp_path / "regions.csv"
    times = tmp_path / "time_in_years.csv"
    regions.write_text("A\nB\n", encoding="utf-8")
    times.write_text("2025\n2026\n", encoding="utf-8")
    return regions, times


def _write_projection(path: Path, rows: list[dict], sheet_name: str = "PkBudg1000_q_ij") -> None:
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)


def test_copy_trade_writes_cs4r_files(tmp_path):
    regions, _ = _write_dimensions(tmp_path)
    projection = tmp_path / "future.xlsx"
    _write_projection(
        projection,
        [{"i": "A", "j": "B", "year": year, "quantity": 2.0} for year in (2025, 2026)]
        + [{"i": "B", "j": "A", "year": year, "quantity": 3.0} for year in (2025, 2026)],
    )

    imports_path, exports_path = copy_trade_to_mfa(
        model=ModelNames.STEEL,
        source=projection,
        scenario_pkbudg=AtlasScenarioPkBudg.BUDG_1000,
        input_data_path=tmp_path / "mfa",
        region_dimension_path=regions,
    )

    assert imports_path.read_text(encoding="utf-8") == (
        "* note: dimensions: (Time,Region,value)\n2025,A,3\n2025,B,2\n2026,A,3\n2026,B,2\n"
    )
    assert exports_path.read_text(encoding="utf-8") == (
        "* note: dimensions: (Time,Region,value)\n2025,A,2\n2025,B,3\n2026,A,2\n2026,B,3\n"
    )


def test_copy_trade_rejects_domestic_flows(tmp_path):
    regions, _ = _write_dimensions(tmp_path)
    projection = tmp_path / "future.xlsx"
    _write_projection(
        projection,
        [
            {"i": "A", "j": "B", "year": 2025, "quantity": 2.0},
            {"i": "A", "j": "A", "year": 2025, "quantity": 2.0},
        ],
    )

    with pytest.raises(AtlasCouplingError, match="domestic"):
        copy_trade_to_mfa(
            model=ModelNames.STEEL,
            source=projection,
            scenario_pkbudg=AtlasScenarioPkBudg.BUDG_1000,
            input_data_path=tmp_path / "mfa",
            region_dimension_path=regions,
        )


def test_cement_is_an_unsupported_model():
    with pytest.raises(AtlasCouplingError, match="does not support"):
        get_model_spec("cement")


def test_atlas_scenario_tag_is_material_specific():
    assert get_model_spec(ModelNames.STEEL).atlas_scenario_tag is None
    assert get_model_spec(ModelNames.PLASTICS).atlas_scenario_tag == "Baseline"


def test_scenario_pkbudg_is_effective_only_for_steel():
    assert scenario_pkbudg_is_effective(ModelNames.STEEL) is True
    assert scenario_pkbudg_is_effective(ModelNames.PLASTICS) is False


def test_load_atlas_trade_projection_uses_baseline_sheet_for_plastics(tmp_path):
    regions, _ = _write_dimensions(tmp_path)
    projection = tmp_path / "future.xlsx"
    _write_projection(
        projection,
        [{"i": "A", "j": "B", "year": 2025, "quantity": 2.0}],
        sheet_name="Baseline_q_ij",
    )

    # scenario_pkbudg is deliberately non-default here to prove plastics ignores it.
    data = load_atlas_trade_projection(
        ModelNames.PLASTICS, projection, AtlasScenarioPkBudg.BUDG_650, regions
    )

    assert data["quantity"].tolist() == [2.0]


def test_copy_trade_writes_cs4r_files_for_plastics(tmp_path, monkeypatch):
    _write_historic_plastics_trade(
        tmp_path,
        monkeypatch,
        imports="2024,A,Plastics,LDPE,1\n2024,B,Plastics,LDPE,1\n",
        exports="2024,A,Plastics,LDPE,1\n2024,B,Plastics,LDPE,1\n",
    )

    regions, _ = _write_dimensions(tmp_path)
    projection = tmp_path / "future.xlsx"
    _write_projection(
        projection,
        [{"i": "A", "j": "B", "year": year, "quantity": 2.0} for year in (2025, 2026)]
        + [{"i": "B", "j": "A", "year": year, "quantity": 3.0} for year in (2025, 2026)],
        sheet_name="Baseline_q_ij",
    )

    imports_path, exports_path = copy_trade_to_mfa(
        model=ModelNames.PLASTICS,
        source=projection,
        # A non-default scenario_pkbudg is deliberately passed to prove plastics ignores it
        # and still reads the "Baseline_q_ij" sheet instead of failing with
        # "Worksheet named 'PkBudg650_q_ij' not found".
        scenario_pkbudg=AtlasScenarioPkBudg.BUDG_650,
        input_data_path=tmp_path / "mfa",
        region_dimension_path=regions,
    )

    assert imports_path.read_text(encoding="utf-8") == (
        "* note: dimensions: (Time,Region,Type,Material,value)\n"
        "2025,A,Plastics,LDPE,3.0\n2025,B,Plastics,LDPE,2.0\n"
        "2026,A,Plastics,LDPE,3.0\n2026,B,Plastics,LDPE,2.0\n"
    )
    assert exports_path.read_text(encoding="utf-8") == (
        "* note: dimensions: (Time,Region,Type,Material,value)\n"
        "2025,A,Plastics,LDPE,2.0\n2025,B,Plastics,LDPE,3.0\n"
        "2026,A,Plastics,LDPE,2.0\n2026,B,Plastics,LDPE,3.0\n"
    )
