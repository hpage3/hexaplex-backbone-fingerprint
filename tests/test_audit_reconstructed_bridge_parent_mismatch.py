from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_reconstructed_bridge_parent_mismatch import (
    build_mismatch_rows,
    ca_counts_by_chain,
    composition_summary,
    parse_pdb_atoms,
    residue_counts_by_chain,
    resolve_reconstructed_path,
    run_audit,
)


def atom_line(serial: int, name: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


def write_fixture_pdb(path: Path, chains: list[str], residues: list[str], include_carboxylate: bool = False) -> None:
    lines = []
    serial = 1
    for chain_index, chain in enumerate(chains):
        for resseq, resname in enumerate(residues, start=1):
            base_x = float(chain_index * 5)
            base_z = float(resseq)
            for name, element, dx in [("N", "N", 0.0), ("CA", "C", 1.0), ("C", "C", 2.0), ("O", "O", 3.0)]:
                lines.append(atom_line(serial, name, resname, chain, resseq, base_x + dx, float(chain_index), base_z, element))
                serial += 1
            if include_carboxylate:
                lines.append(atom_line(serial, "OE1", resname, chain, resseq, base_x + 4.0, float(chain_index), base_z, "O"))
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_pdb_composition_parsing_works(tmp_path: Path) -> None:
    pdb = tmp_path / "model.pdb"
    write_fixture_pdb(pdb, ["A", "B"], ["GLU", "CYP"], include_carboxylate=True)

    atoms = parse_pdb_atoms(pdb)
    summary = composition_summary(atoms)

    assert summary["atom_count"] == 20
    assert summary["heavy_atom_count"] == 20
    assert summary["chain_count"] == 2
    assert summary["residue_count"] == 4
    assert summary["carboxylate_present"] is True
    assert summary["peptide_backbone_atoms_present"] is True


def test_residue_atom_chain_summaries_are_produced(tmp_path: Path) -> None:
    pdb = tmp_path / "model.pdb"
    write_fixture_pdb(pdb, ["A", "B"], ["PPI", "PPJ"], include_carboxylate=False)
    atoms = parse_pdb_atoms(pdb)

    assert residue_counts_by_chain(atoms) == {"A": 2, "B": 2}
    assert ca_counts_by_chain(atoms) == {"A": 2, "B": 2}


def test_mismatch_severity_for_atom_and_chain_count_differences(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    recon = tmp_path / "recon.pdb"
    write_fixture_pdb(parent, ["A", "B"], ["GLU", "CYP"], include_carboxylate=True)
    write_fixture_pdb(recon, ["A"], ["PPI", "PPJ"], include_carboxylate=False)

    summary = build_mismatch_rows(parse_pdb_atoms(parent), parse_pdb_atoms(recon))
    rows = summary.set_index("metric")

    assert rows.loc["atom_count", "severity"] == "high"
    assert rows.loc["chain_count", "severity"] == "high"
    assert rows.loc["carboxylate_present", "severity"] == "high"


def test_resolve_reconstructed_file_from_scores_csv(tmp_path: Path) -> None:
    recon = tmp_path / "reconstructed.pdb"
    write_fixture_pdb(recon, ["A"], ["PPI"])
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [{"variant_id": "reconstructed_rise_3p40", "coordinate_path": str(recon)}]
    ).to_csv(scores, index=False)

    resolved = resolve_reconstructed_path(tmp_path / "missing.pdb", scores)

    assert resolved == recon


def test_missing_reconstructed_file_is_handled_clearly(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    write_fixture_pdb(parent, ["A"], ["GLU"])

    with pytest.raises(FileNotFoundError, match="Missing reconstructed PDB"):
        run_audit(
            parent_pdb=parent,
            reconstructed_pdb=tmp_path / "missing.pdb",
            bridge_scores_csv=tmp_path / "missing_scores.csv",
            summary_csv=tmp_path / "summary.csv",
            peak_csv=tmp_path / "peaks.csv",
            report_path=tmp_path / "report.md",
        )


def test_output_csv_and_report_are_written(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pdb"
    recon = tmp_path / "recon.pdb"
    write_fixture_pdb(parent, ["A", "B"], ["GLU", "CYP"], include_carboxylate=True)
    write_fixture_pdb(recon, ["A", "B"], ["PPI", "PPJ"], include_carboxylate=False)
    summary_csv = tmp_path / "summary.csv"
    peak_csv = tmp_path / "peaks.csv"
    report = tmp_path / "report.md"

    summary, peak_summary, diagnosis = run_audit(
        parent_pdb=parent,
        reconstructed_pdb=recon,
        bridge_scores_csv=tmp_path / "scores.csv",
        summary_csv=summary_csv,
        peak_csv=peak_csv,
        report_path=report,
    )

    assert summary_csv.exists()
    assert peak_csv.exists()
    assert report.exists()
    assert not summary.empty
    assert not peak_summary.empty
    assert diagnosis["bridge_status"] in {"not_parent_equivalent", "unresolved"}
    assert "not parent-equivalent" in report.read_text(encoding="utf-8")
