def test_import_modules():
    import hexaplex_backbone_fingerprint.band_mapping
    import hexaplex_backbone_fingerprint.geometry
    import hexaplex_backbone_fingerprint.io
    import hexaplex_backbone_fingerprint.pdb_parser
    import hexaplex_backbone_fingerprint.peptide_planes

    assert hexaplex_backbone_fingerprint.geometry is not None
