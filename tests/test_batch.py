"""Pytest suite for Q-Med shared run_molecule_pipeline and Batch Screening."""

from pathlib import Path
import time
from typing import Any, Dict

from app import build_batch_decisions_chart
from qmed.batch import (
    BatchConfig,
    get_batch_items,
    get_sample_batch_csv,
    list_batches,
    process_batch_row,
    run_batch_screening,
    validate_batch_csv,
)
from qmed.config import BATCH_CUSTOM_GEOMETRY_CAPTION
from qmed.storage import (
    MoleculePipelineConfig,
    cap_active_space,
    run_molecule_pipeline,
)


def test_shared_run_molecule_pipeline_real_integration_h2_and_f() -> None:
    """Real integration test (no mocks): run_molecule_pipeline on '[H][H]' and 'F' (ideal only) within 1 mHa of matched reference."""
    cfg = MoleculePipelineConfig(
        seeds=(42, 43, 44),
        geometry_seed=42,
        capped_active_space=(4, 4),
        run_noisy=False,
        ideal_maxiter=35,
        persist_to_sqlite=False,
    )

    # 1. Real integration run on '[H][H]'
    res_h2 = run_molecule_pipeline("[H][H]", config=cfg)
    assert res_h2.qee.decision == "SIMULATED_QUANTUM"
    assert res_h2.outcome == "SIMULATED"
    assert res_h2.status_category == "SIMULATED"
    assert "simulated" in res_h2.status_message.lower()
    assert "skipped" not in res_h2.status_message.lower()
    assert res_h2.effective_active_space == (2, 2)
    assert res_h2.ideal_benchmark is not None
    assert res_h2.ideal_vqe_energy is not None
    assert res_h2.matched_fci_energy is not None
    assert res_h2.ideal_error_mha is not None
    assert abs(res_h2.ideal_vqe_energy - res_h2.matched_fci_energy) * 1000.0 <= 1.0
    assert res_h2.ideal_error_mha <= 1.0
    assert res_h2.geometry_caption == BATCH_CUSTOM_GEOMETRY_CAPTION

    # 2. Real integration run on 'F' (Hydrogen Fluoride, 8e in 5 valence orbitals: 4 occ + 1 vir -> capped to (4e, 3o))
    assert cap_active_space((4, 4), available_electrons=8, available_orbitals=5) == (4, 3)
    res_f = run_molecule_pipeline("F", config=cfg)
    assert res_f.qee.decision == "SIMULATED_QUANTUM"
    assert res_f.outcome == "SIMULATED"
    assert res_f.status_category == "SIMULATED"
    assert "simulated" in res_f.status_message.lower()
    assert "skipped" not in res_f.status_message.lower()
    assert res_f.effective_active_space == (4, 3)
    assert res_f.ideal_benchmark is not None
    assert res_f.ideal_vqe_energy is not None
    assert res_f.matched_fci_energy is not None
    assert res_f.ideal_error_mha is not None
    assert abs(res_f.ideal_vqe_energy - res_f.matched_fci_energy) * 1000.0 <= 1.0
    assert res_f.ideal_error_mha <= 1.0
    assert res_f.active_space_error_mha is not None
    assert res_f.active_space_error_mha > 1.6
    # Because active-space truncation error exceeds 1.6 mHa, reliability is capped at MEDIUM
    assert res_f.reliability == "MEDIUM"
    assert res_f.geometry_caption == BATCH_CUSTOM_GEOMETRY_CAPTION


def test_csv_validation_valid_invalid_duplicates_and_row_limit() -> None:
    """Verify CSV validation handles valid rows, duplicates removal, invalid SMILES, and row limit."""
    sample_csv = get_sample_batch_csv()
    res = validate_batch_csv(sample_csv, max_rows=500)

    assert res.is_valid is True
    assert res.total_rows == 10
    # 1 duplicate ('Water_Duplicate,O,0,0') and 1 invalid ('Malformed_Input,not_a_valid_smiles,0,0')
    assert res.duplicates_removed == 1
    assert len(res.duplicate_rows) == 1
    assert res.duplicate_rows[0]["canonical_smiles"] == "O"
    assert len(res.invalid_rows) == 1
    assert "not_a_valid_smiles" in res.invalid_rows[0].input_smiles
    assert "Invalid SMILES" in res.invalid_rows[0].reason
    assert len(res.valid_rows) == 8

    # Missing 'smiles' column must fail validation cleanly
    bad_header_res = validate_batch_csv("molecule_name,formula\nWater,H2O\n")
    assert bad_header_res.is_valid is False
    assert "smiles" in (bad_header_res.error or "").lower()

    # Exceeding max_rows limit must fail validation cleanly
    over_limit_res = validate_batch_csv(sample_csv, max_rows=3)
    assert over_limit_res.is_valid is False
    assert "exceeds the maximum limit" in (over_limit_res.error or "")


def test_process_batch_row_rejections_and_status_categories() -> None:
    """Verify invalid SMILES, charged, open-shell, unsupported-element, DEFERRED, and CLASSICAL_ONLY status messages."""
    # 1. Invalid SMILES
    inv_item = process_batch_row("not_a_smiles_string", name="BadMol")
    assert inv_item.qee_decision == "REJECTED"
    assert inv_item.outcome == "REJECTED_INVALID_SMILES"
    assert any("Invalid SMILES" in r for r in inv_item.reasons)

    # 2. Charged species ([OH-] or charge != 0)
    charged_item = process_batch_row("[OH-]", name="Hydroxide", charge=-1, spin=0)
    assert charged_item.qee_decision == "REJECTED"
    assert charged_item.outcome == "REJECTED_CHARGED"
    assert any("Charged molecule" in r for r in charged_item.reasons)

    # 3. Open-shell radical ([CH3] or spin != 0)
    open_shell_item = process_batch_row("[CH3]", name="MethylRadical", charge=0, spin=1)
    assert open_shell_item.qee_decision == "REJECTED"
    assert open_shell_item.outcome == "REJECTED_OPEN_SHELL"
    assert any("Open-shell" in r for r in open_shell_item.reasons)

    # 4. Unsupported element (Cl in chloroethane)
    unsup_item = process_batch_row("CCCl", name="Chloroethane", charge=0, spin=0)
    assert unsup_item.qee_decision == "REJECTED"
    assert unsup_item.outcome == "REJECTED_UNSUPPORTED_ELEMENT"
    assert any("Cl" in r for r in unsup_item.reasons)

    # 5. Status categories on shared pipeline for CLASSICAL_ONLY and DEFERRED
    he_pipe = run_molecule_pipeline("[He]")
    assert he_pipe.status_category == "CLASSICAL_ONLY"
    assert "CLASSICAL_ONLY" in he_pipe.status_message

    asp_pipe = run_molecule_pipeline("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert asp_pipe.status_category == "DEFERRED"
    assert "DEFERRED" in asp_pipe.status_message


def test_process_batch_row_timeout_and_completed_with_errors_tracking(tmp_path: Path) -> None:
    """Verify real per-molecule timeout captures real stage/exception and sets batch status to COMPLETED_WITH_ERRORS."""
    cfg_timeout = BatchConfig(
        seeds=(42,),
        ideal_maxiter=5,
        per_molecule_timeout_sec=0.001,
    )
    timeout_item = process_batch_row(
        "[H][H]",
        name="SlowHydrogen",
        config=cfg_timeout,
        run_simulation=True,
    )
    assert timeout_item.qee_decision == "SIMULATED_QUANTUM"
    assert timeout_item.outcome == "TIMEOUT"
    assert timeout_item.error_stage is not None
    assert timeout_item.error_message is not None
    assert "TimeoutError" in timeout_item.error_message

    # Verify batch status is COMPLETED_WITH_ERRORS when a simulation times out in a real batch run
    db_err = tmp_path / "test_completed_with_errors.db"
    csv_two = "name,smiles,charge,spin\nHelium,[He],0,0\nHydrogen,[H][H],0,0\n"
    batch_err_res = run_batch_screening(
        csv_two,
        config=BatchConfig(seeds=(42,), ideal_maxiter=5, per_molecule_timeout_sec=0.001, resume=False),
        db_path=db_err,
    )
    assert batch_err_res.status == "COMPLETED_WITH_ERRORS"
    assert batch_err_res.simulated_count == 0
    assert batch_err_res.failed_count == 1
    h2_item = [it for it in batch_err_res.items if it.name == "Hydrogen"][0]
    assert h2_item.outcome == "TIMEOUT"
    assert h2_item.error_stage is not None
    assert "TimeoutError" in (h2_item.error_message or "")


def test_batch_sqlite_storage_and_resume(tmp_path: Path) -> None:
    """Verify full batch screening saves to batches/batch_items in SQLite and resumes stored rows."""
    db_file = tmp_path / "test_batch_resume.db"
    csv_text = "\n".join(
        [
            "name,smiles,charge,spin",
            "Hydrogen,[H][H],0,0",
            "Helium,[He],0,0",
            "Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,0,0",
            "Hydrogen_Dup,[H][H],0,0",
            "Methyl_Radical,[CH3],0,1",
        ]
    )

    cfg = BatchConfig(
        seeds=(42, 43, 44),
        run_noisy=False,
        ideal_maxiter=15,
        max_simulated_molecules=5,
        resume=True,
    )

    # First batch run: simulates Hydrogen and screens Helium, Aspirin, and rejects Methyl_Radical
    run1 = run_batch_screening(csv_text, config=cfg, db_path=db_file)
    assert run1.status == "COMPLETED"
    assert run1.validation.duplicates_removed == 1
    assert len(run1.items) == 4
    assert run1.simulated_count == 1
    assert run1.failed_count == 0
    assert run1.resumed_count == 0

    df1 = run1.to_dataframe()
    assert len(df1) == 4
    assert "Error Stage" in df1.columns
    assert "Error Message" in df1.columns
    fig = build_batch_decisions_chart(run1)
    assert len(fig.data) == 1

    # Second batch run with identical config and seed: Hydrogen, Helium, and Aspirin must resume from SQLite
    run2 = run_batch_screening(csv_text, config=cfg, db_path=db_file)
    assert run2.status == "COMPLETED"
    assert run2.resumed_count == 3  # Hydrogen (SIMULATED), Helium (SCREENED_ONLY), Aspirin (SCREENED_ONLY)
    assert run2.simulated_count == 0
    assert run2.failed_count == 0

    h2_resumed = [it for it in run2.items if it.name == "Hydrogen"][0]
    assert h2_resumed.was_resumed is True
    assert h2_resumed.outcome == "RESUMED"
    assert h2_resumed.ideal_vqe_energy == run1.items[0].ideal_vqe_energy

    batches = list_batches(db_path=db_file)
    assert len(batches) == 2
    items_b1 = get_batch_items(run1.batch_id, db_path=db_file)
    assert len(items_b1) == 4
