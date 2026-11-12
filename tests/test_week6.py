"""Week 6 pytest suite: 3D coordinate/bond/angle pure functions, py3Dmol/Plotly viewers, glossary, error chart, and CI/README checks."""

import json
from pathlib import Path
import pytest

from app import (
    DECISION_COLORS,
    PLAIN_LANGUAGE_GLOSSARY,
    PRESET_MOLECULES,
    build_error_vs_exact_figure,
    build_geometry_tables,
    build_plotly_3d_molecule_figure,
    build_py3dmol_html,
    build_results_dataframe,
    format_badge_html,
    render_rdkit_2d_image,
)
from qmed.config import PROJECT_ROOT
from qmed.molecules import (
    APPROXIMATE_GEOMETRY_CAPTION,
    MoleculeValidationError,
    compute_angle_degrees,
    compute_bond_lengths_and_angles,
    compute_distance_angstrom,
    load_benchmarks,
    load_molecule_3d_structure,
    smiles_to_3d_coordinates,
)
from qmed.storage import run_qmed_pipeline


def test_benchmark_3d_coordinates_and_bond_lengths_angles() -> None:
    """Verify exact benchmark coordinates from data/benchmarks.json and bond lengths/angles."""
    raw_benchmarks = load_benchmarks()

    # 1. H2: bond length = 0.7350 A, 0 bond angles
    h2_struct = load_molecule_3d_structure("H2", use_benchmark_coords=True)
    assert h2_struct.source == "benchmark_exact"
    assert h2_struct.atoms == raw_benchmarks["H2"].atoms
    assert len(h2_struct.bond_lengths) == 1
    assert h2_struct.bond_lengths[0].bond_label == "H1–H2"
    assert h2_struct.bond_lengths[0].distance_angstrom == pytest.approx(0.7350, abs=1e-4)
    assert len(h2_struct.bond_angles) == 0

    # 2. LiH: bond length = 1.5950 A, 0 bond angles
    lih_struct = load_molecule_3d_structure("LiH", use_benchmark_coords=True)
    assert lih_struct.source == "benchmark_exact"
    assert lih_struct.atoms == raw_benchmarks["LiH"].atoms
    assert len(lih_struct.bond_lengths) == 1
    assert lih_struct.bond_lengths[0].bond_label == "Li1–H2"
    assert lih_struct.bond_lengths[0].distance_angstrom == pytest.approx(1.5950, abs=1e-4)
    assert len(lih_struct.bond_angles) == 0

    # 3. H2O: two O-H bonds = 0.9572 A, one H-O-H bond angle = 104.52 degrees
    h2o_struct = load_molecule_3d_structure("H2O", use_benchmark_coords=True)
    assert h2o_struct.source == "benchmark_exact"
    assert h2o_struct.atoms == raw_benchmarks["H2O"].atoms
    assert len(h2o_struct.bond_lengths) == 2
    for bl in h2o_struct.bond_lengths:
        assert bl.distance_angstrom == pytest.approx(0.9572, abs=1e-4)
    assert len(h2o_struct.bond_angles) == 1
    assert h2o_struct.bond_angles[0].angle_label == "H2–O1–H3"
    assert h2o_struct.bond_angles[0].angle_degrees == pytest.approx(104.52, abs=0.05)


def test_smiles_to_3d_mmff_and_caption() -> None:
    """Verify RDKit SMILES-to-3D (AddHs, seeded EmbedMolecule, MMFF) and approximate caption."""
    atoms1, bonds1, can1, form1 = smiles_to_3d_coordinates("CCO", seed=42)
    atoms2, bonds2, _, _ = smiles_to_3d_coordinates("CCO", seed=42)

    assert can1 == "CCO"
    assert form1 == "C2H6O"
    assert len(atoms1) == 9  # 2 C + 1 O + 6 H
    assert len(bonds1) == 8
    # Fixed seed must reproduce identical 3D coordinates
    assert atoms1 == atoms2
    assert bonds1 == bonds2

    lengths, angles = compute_bond_lengths_and_angles(atoms1, bonds=bonds1)
    assert len(lengths) == 8
    assert len(angles) > 0

    # Custom SMILES structure must carry the exact required force-field disclaimer caption
    custom_struct = load_molecule_3d_structure("CCO", use_benchmark_coords=False, seed=42)
    assert custom_struct.source == "rdkit_mmff_approx"
    assert custom_struct.caption == APPROXIMATE_GEOMETRY_CAPTION

    # Monoatomic noble gas [He] works cleanly with 1 atom and 0 bonds
    he_struct = load_molecule_3d_structure("[He]", use_benchmark_coords=False, seed=42)
    assert len(he_struct.atoms) == 1
    assert len(he_struct.bond_lengths) == 0

    # Invalid SMILES raises MoleculeValidationError cleanly
    with pytest.raises(MoleculeValidationError):
        smiles_to_3d_coordinates("not_a_valid_smiles_string")


def test_pure_distance_and_angle_math() -> None:
    """Verify pure Euclidean distance and 3-point angle math."""
    d = compute_distance_angstrom([0.0, 0.0, 0.0], [0.0, 3.0, 4.0])
    assert d == pytest.approx(5.0, abs=1e-12)

    ang_90 = compute_angle_degrees([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert ang_90 == pytest.approx(90.0, abs=1e-9)

    ang_180 = compute_angle_degrees([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert ang_180 == pytest.approx(180.0, abs=1e-9)


def test_py3dmol_html_plotly_fallback_and_2d_draw() -> None:
    """Verify py3Dmol _make_html(), Plotly 3D fallback, 2D RDKit image, and geometry tables."""
    h2o_struct = load_molecule_3d_structure("H2O", use_benchmark_coords=True)

    for style in ["Ball-and-stick", "Stick", "Space-filling", "Wireframe"]:
        html_out = build_py3dmol_html(
            h2o_struct,
            style=style,
            show_labels=True,
            show_hydrogens=False,
            spin=True,
            background="Dark",
        )
        assert isinstance(html_out, str)
        assert "3Dmol" in html_out or "viewer" in html_out

    fig_3d = build_plotly_3d_molecule_figure(
        h2o_struct, show_labels=True, show_hydrogens=True, background="Light"
    )
    assert len(fig_3d.data) >= 3  # 2 O-H bond line traces + 1 atom markers trace

    img_2d = render_rdkit_2d_image("O")
    assert img_2d is not None

    df_bonds, df_angles = build_geometry_tables(h2o_struct)
    assert len(df_bonds) == 2
    assert len(df_angles) == 1
    assert "104.52°" in df_angles["Degrees (°)"].values


def test_plain_language_glossary_and_error_vs_exact_chart(tmp_path: Path) -> None:
    """Verify glossary terms and the Error-vs-Exact (mHa) chart with 1.6 mHa chemical accuracy line."""
    required_terms = {
        "SMILES",
        "Hartree",
        "mHa",
        "HF",
        "FCI/CASCI",
        "VQE",
        "UCCSD",
        "qubit",
        "seed",
        "shots",
        "noise",
        "active space",
        "reliability labels",
        "quantum advantage",
    }
    assert required_terms.issubset(set(PLAIN_LANGUAGE_GLOSSARY.keys()))

    badge = format_badge_html("SIMULATED_QUANTUM", DECISION_COLORS, tooltip="test tooltip")
    assert "SIMULATED_QUANTUM" in badge
    assert "test tooltip" in badge

    db_file = tmp_path / "test_week6_ui.db"
    pipe_res = run_qmed_pipeline(
        "H2",
        seeds=(42, 43, 44),
        run_noisy=True,
        shots=512,
        noisy_maxiter=20,
        db_path=db_file,
    )

    df = build_results_dataframe(pipe_res)
    assert len(df) == 4  # Classical HF, Classical FCI, VQE Ideal, VQE Noisy
    assert "Mean Energy (Ha)" in df.columns

    fig = build_error_vs_exact_figure(pipe_res)
    assert len(fig.data) == 1
    bar_trace = fig.data[0]
    assert len(bar_trace.x) == 3  # Classical HF, VQE Ideal, VQE Noisy
    assert bar_trace.error_y is not None
    assert len(bar_trace.error_y.array) == 3
    # Verify horizontal chemical accuracy line at 1.6 mHa is present
    assert len(fig.layout.shapes) >= 1
    assert fig.layout.shapes[0].y0 == pytest.approx(1.6, abs=1e-6)


def test_h2_curve_notebook_and_png_artifact() -> None:
    """Verify notebooks/h2_curve.ipynb is valid nbformat v4 JSON and docs/h2_curve.png exists."""
    nb_path = PROJECT_ROOT / "notebooks" / "h2_curve.ipynb"
    assert nb_path.exists()

    with nb_path.open("r", encoding="utf-8") as handle:
        nb_data = json.load(handle)

    assert nb_data.get("nbformat") == 4
    cells = nb_data.get("cells", [])
    assert len(cells) >= 3
    code_sources = "\n".join(
        "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
    )
    assert "run_h2_bond_scan" in code_sources
    assert "save_h2_curve_plot" in code_sources
    assert "h2_curve.png" in code_sources

    png_path = PROJECT_ROOT / "docs" / "h2_curve.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 10_000


def test_github_actions_workflow_and_readme_completeness() -> None:
    """Verify .github/workflows/ci.yml and README.md contain required sections and real results."""
    ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists()
    ci_text = ci_path.read_text(encoding="utf-8")
    assert "pytest" in ci_text
    assert "requirements.txt" in ci_text

    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists()
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "-1.137306" in readme_text
    assert "-7.882175" in readme_text
    assert "-74.970341" in readme_text
    assert "NO_DEMONSTRATED_ADVANTAGE" in readme_text
    assert "Limitations" in readme_text
    assert "Streamlit Community Cloud" in readme_text
