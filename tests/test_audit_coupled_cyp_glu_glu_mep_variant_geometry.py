from scripts.audit_coupled_cyp_glu_glu_mep_variant_geometry import classify_geometry_metrics


def clean_metrics() -> dict[str, object]:
    return {
        "candidate_file_exists": True,
        "atom_count_match": True,
        "labels_preserved": True,
        "max_ca_shift_A": 0.0,
        "max_backbone_bond_delta_A": 0.01,
        "max_backbone_angle_delta_deg": 1.0,
        "max_omega_trans_deviation_deg": 10.0,
    }


def test_geometry_threshold_classifier_passes_clean_row() -> None:
    assert classify_geometry_metrics(clean_metrics()) == (True, "")


def test_geometry_threshold_classifier_fails_bond_delta() -> None:
    metrics = clean_metrics()
    metrics["max_backbone_bond_delta_A"] = 0.051
    assert classify_geometry_metrics(metrics) == (False, "backbone_bond_delta_exceeds_threshold")


def test_geometry_threshold_classifier_fails_angle_delta() -> None:
    metrics = clean_metrics()
    metrics["max_backbone_angle_delta_deg"] = 5.1
    assert classify_geometry_metrics(metrics) == (False, "backbone_angle_delta_exceeds_threshold")


def test_geometry_threshold_classifier_fails_omega_delta() -> None:
    metrics = clean_metrics()
    metrics["max_omega_trans_deviation_deg"] = 15.1
    assert classify_geometry_metrics(metrics) == (False, "omega_trans_deviation_exceeds_threshold")


def test_failed_checks_formatting_is_stable_for_multiple_failures() -> None:
    metrics = clean_metrics()
    metrics["max_backbone_bond_delta_A"] = 0.2
    metrics["max_omega_trans_deviation_deg"] = 20.0
    assert classify_geometry_metrics(metrics)[1] == "backbone_bond_delta_exceeds_threshold;omega_trans_deviation_exceeds_threshold"
