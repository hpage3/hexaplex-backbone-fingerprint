from pathlib import Path

from scripts.generate_coupled_cyp_glu_glu_mep_variants import (
    CYP_OMEGA_MODE,
    GLU_OMEGA_MODE,
    coupled_delta_grid,
    manifest_row,
    output_path,
    variant_id,
)


def test_coupled_delta_grid_has_six_combinations_and_baseline() -> None:
    grid = coupled_delta_grid()
    assert len(grid) == 6
    assert (0.0, 0.0) in grid


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(-1.0, 0.0) == "cyp_glu_m1__glu_mep_p0"
    assert variant_id(1.0, -1.0) == "cyp_glu_p1__glu_mep_m1"


def test_output_path_is_under_coupled_directory() -> None:
    path = output_path(Path("outputs/coordinates/coupled_cyp_glu_glu_mep_variants"), "v")
    assert path.as_posix().endswith("outputs/coordinates/coupled_cyp_glu_glu_mep_variants/v.pdb")


def test_manifest_row_includes_both_deltas_and_omega_modes(tmp_path: Path) -> None:
    row = manifest_row("v", -1.0, 0.0, tmp_path / "source.pdb", tmp_path / "v.pdb", 45, 45, [], 42, 42, [], 0.0)
    assert row["cyp_glu_delta_deg"] == -1.0
    assert row["glu_mep_delta_deg"] == 0.0
    assert row["cyp_glu_omega_mode"] == CYP_OMEGA_MODE
    assert row["glu_mep_omega_mode"] == GLU_OMEGA_MODE
    assert row["status"] == "ok"
