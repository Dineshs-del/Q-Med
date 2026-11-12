"""Week 6 pytest suite: Streamlit app helpers, h2_curve.ipynb, GitHub Actions CI, and README verification."""

import json
from pathlib import Path

from app import (
    DECISION_COLORS,
    PRESET_MOLECULES,
    build_ideal_vs_noisy_figure,
    build_results_dataframe,
    format_badge_html,
)
from qmed.config import PROJECT_ROOT
from qmed.storage import run_qmed_pipeline


def test_streamlit_app_helpers_and_presets(tmp_path: Path) -> None:
    """Verify app.py helper functions build results DataFrame and Plotly chart from live pipeline output."""
    assert "H2" in PRESET_MOLECULES.values()
    assert "LiH" in PRESET_MOLECULES.values()
    assert "H2O" in PRESET_MOLECULES.values()

    badge = format_badge_html("SIMULATED_QUANTUM", DECISION_COLORS)
    assert "SIMULATED_QUANTUM" in badge
    assert "#16a34a" in badge

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
    assert "Reliability" in df.columns
    assert "Advantage Verdict" in df.columns

    fig = build_ideal_vs_noisy_figure(pipe_res)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 4


def test_h2_curve_notebook_and_png_artifact() -> None:
    """Verify notebooks/h2_curve.ipynb is valid nbformat v4 JSON and docs/h2_curve.png exists."""
    nb_path = PROJECT_ROOT / "notebooks" / "h2_curve.ipynb"
    assert nb_path.exists()

    with nb_path.open("r", encoding="utf-8") as handle:
        nb_data = json.load(handle)

    assert nb_data.get("nbformat") == 4
    cells = nb_data.get("cells", [])
    assert len(cells) >= 3
    code_sources = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")
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
