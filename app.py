"""Streamlit decision-support UI for Q-Med: Quantum Eligibility, VQE vs PySCF FCI, and SQLite History."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qmed.config import (
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_DB_PATH,
    DEFAULT_SHOTS,
    MAX_ACTIVE_SPACE_ERROR_FOR_HIGH_MHA,
    MAX_CIRCUIT_DEPTH,
    MAX_HYBRID_FULL_QUBITS,
    MAX_SIMULATED_QUBITS,
    PROJECT_ROOT,
)
from qmed.molecules import validate_and_describe
from qmed.qee import QEEResult, evaluate_eligibility, resolve_input_smiles
from qmed.storage import (
    PipelineExecutionResult,
    list_experiments,
    replay_experiment,
    run_qmed_pipeline,
)
from qmed.vqe_runner import run_water_option_b_demo

PRESET_MOLECULES: Dict[str, str] = {
    "H2 — Hydrogen (0.735 Å Benchmark)": "H2",
    "LiH — Lithium Hydride (1.595 Å Benchmark)": "LiH",
    "H2O — Water (Equilibrium Benchmark)": "H2O",
    "Aspirin — Acetylsalicylic Acid (DEFERRED Demo)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Helium — Closed-Shell Atom (CLASSICAL_ONLY Demo)": "[He]",
    "Ammonia — NH3 (HYBRID Active-Space Demo)": "N",
    "Ethanol — C2H5OH (DEFERRED Demo)": "CCO",
}

DECISION_COLORS: Dict[str, str] = {
    "SIMULATED_QUANTUM": "#16a34a",
    "HYBRID": "#d97706",
    "CLASSICAL_ONLY": "#2563eb",
    "DEFERRED": "#dc2626",
}

RELIABILITY_COLORS: Dict[str, str] = {
    "HIGH": "#16a34a",
    "MEDIUM": "#ca8a04",
    "LOW": "#ea580c",
    "UNRELIABLE": "#dc2626",
    "INVALID": "#7f1d1d",
}


def format_badge_html(label: str, color_map: Dict[str, str]) -> str:
    """Render an inline HTML badge for QEE decisions, reliability labels, or verdicts."""
    bg = color_map.get(label, "#475569")
    return (
        f'<span style="background-color:{bg};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:600;font-size:0.85rem;">{label}</span>'
    )


def build_results_dataframe(result: PipelineExecutionResult) -> pd.DataFrame:
    """Build a comparison DataFrame of Classical HF, Exact FCI/CASCI, Ideal VQE, and Noisy VQE."""
    rows: List[Dict[str, Any]] = []
    if result.ideal_benchmark is None:
        return pd.DataFrame(rows)

    ideal = result.ideal_benchmark
    space_str = f"({ideal.active_electrons}e, {ideal.active_orbitals}o)"

    # 1. Classical Hartree-Fock baseline
    hf_err_mha = abs(ideal.hf_energy - ideal.matched_fci_energy) * 1000.0
    rows.append(
        {
            "Method": "Classical Hartree–Fock (PySCF)",
            "Mode": "exact_mean_field",
            "Active Space": space_str,
            "Qubits": ideal.num_qubits,
            "Depth / Gates": "—",
            "Mean Energy (Ha)": round(ideal.hf_energy, 8),
            "Matched FCI/CASCI (Ha)": round(ideal.matched_fci_energy, 8),
            "|Mean Error| (mHa)": round(hf_err_mha, 4),
            "Uncertainty ±σ (mHa)": 0.0,
            "AS Truncation Error (mHa)": round(ideal.active_space_error_mha, 4),
            "Converged": "100%",
            "Mean Runtime (s)": round(ideal.classical_runtime_sec, 4),
            "Reliability": "BASELINE",
            "Advantage Verdict": "BASELINE",
        }
    )

    # 2. Exact FCI / CASCI baseline
    rows.append(
        {
            "Method": "Classical Exact FCI / CASCI (PySCF)",
            "Mode": "exact_diagonalization",
            "Active Space": space_str,
            "Qubits": ideal.num_qubits,
            "Depth / Gates": "—",
            "Mean Energy (Ha)": round(ideal.matched_fci_energy, 8),
            "Matched FCI/CASCI (Ha)": round(ideal.matched_fci_energy, 8),
            "|Mean Error| (mHa)": 0.0,
            "Uncertainty ±σ (mHa)": 0.0,
            "AS Truncation Error (mHa)": round(ideal.active_space_error_mha, 4),
            "Converged": "100%",
            "Mean Runtime (s)": round(ideal.classical_runtime_sec, 4),
            "Reliability": "EXACT_REFERENCE",
            "Advantage Verdict": "BASELINE",
        }
    )

    # 3. Ideal VQE
    rows.append(
        {
            "Method": f"VQE Ideal ({ideal.ansatz} + {ideal.optimizer})",
            "Mode": ideal.noise_mode,
            "Active Space": space_str,
            "Qubits": ideal.num_qubits,
            "Depth / Gates": f"{ideal.circuit_depth} / {ideal.gate_count}",
            "Mean Energy (Ha)": round(ideal.mean_vqe_energy, 8),
            "Matched FCI/CASCI (Ha)": round(ideal.matched_fci_energy, 8),
            "|Mean Error| (mHa)": round(ideal.abs_mean_error_mha, 4),
            "Uncertainty ±σ (mHa)": round(ideal.uncertainty_mha, 4),
            "AS Truncation Error (mHa)": round(ideal.active_space_error_mha, 4),
            "Converged": f"{int(round(ideal.converged_fraction * 100))}% ({len(ideal.seeds)} seeds)",
            "Mean Runtime (s)": round(ideal.mean_vqe_runtime_sec, 3),
            "Reliability": ideal.reliability,
            "Advantage Verdict": ideal.verdict,
        }
    )

    # 4. Noisy VQE (if executed)
    if result.noisy_benchmark is not None:
        noisy = result.noisy_benchmark
        rows.append(
            {
                "Method": f"VQE Noisy ({noisy.ansatz} + {noisy.optimizer})",
                "Mode": f"{noisy.noise_mode} ({noisy.shots} shots)",
                "Active Space": space_str,
                "Qubits": noisy.num_qubits,
                "Depth / Gates": f"{noisy.circuit_depth} / {noisy.gate_count}",
                "Mean Energy (Ha)": round(noisy.mean_vqe_energy, 8),
                "Matched FCI/CASCI (Ha)": round(noisy.matched_fci_energy, 8),
                "|Mean Error| (mHa)": round(noisy.abs_mean_error_mha, 4),
                "Uncertainty ±σ (mHa)": round(noisy.uncertainty_mha, 4),
                "AS Truncation Error (mHa)": round(noisy.active_space_error_mha, 4),
                "Converged": f"{int(round(noisy.converged_fraction * 100))}% ({len(noisy.seeds)} seeds)",
                "Mean Runtime (s)": round(noisy.mean_vqe_runtime_sec, 3),
                "Reliability": noisy.reliability,
                "Advantage Verdict": noisy.verdict,
            }
        )

    return pd.DataFrame(rows)


def build_ideal_vs_noisy_figure(result: PipelineExecutionResult) -> go.Figure:
    """Create a Plotly figure comparing HF, FCI/CASCI, Ideal VQE, and Noisy VQE energies."""
    fig = go.Figure()
    if result.ideal_benchmark is None:
        return fig

    ideal = result.ideal_benchmark
    methods = ["PySCF Exact FCI/CASCI", "PySCF Hartree–Fock", "VQE Ideal (Mean ± σ)"]
    energies = [ideal.matched_fci_energy, ideal.hf_energy, ideal.mean_vqe_energy]
    errors = [0.0, 0.0, ideal.uncertainty_ha]
    colors = ["#0f766e", "#64748b", "#16a34a"]

    if result.noisy_benchmark is not None:
        noisy = result.noisy_benchmark
        methods.append(f"VQE Noisy [{noisy.noise_mode}]")
        energies.append(noisy.mean_vqe_energy)
        errors.append(noisy.uncertainty_ha)
        colors.append("#dc2626")

    fig.add_trace(
        go.Bar(
            x=methods,
            y=energies,
            error_y={"type": "data", "array": errors, "visible": True},
            marker_color=colors,
            text=[f"{e:.6f} Ha" for e in energies],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{ideal.molecule_id} ({ideal.active_electrons}e, {ideal.active_orbitals}o) — Total Energy Comparison (Ha)",
        yaxis_title="Total Energy (Hartree)",
        xaxis_title="Method",
        template="plotly_white",
        height=430,
    )
    return fig


def render_qee_card(qee: QEEResult) -> None:
    """Render the Quantum Eligibility Engine (QEE v1) decision card with reasons and warnings."""
    st.subheader("Quantum Eligibility Engine (QEE v1) Decision")
    decision_label = qee.decision or "INVALID_INPUT"
    badge_html = format_badge_html(decision_label, DECISION_COLORS)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"**Routing Decision**<br>{badge_html}", unsafe_allow_html=True)
    col2.metric(
        "Estimated Qubits (Frozen-Core)",
        f"{qee.features.get('estimated_qubits', '—')} / {MAX_SIMULATED_QUBITS}",
    )
    col3.metric(
        "Estimated UCCSD Depth",
        f"{qee.features.get('estimated_circuit_depth', 0):,} / {MAX_CIRCUIT_DEPTH:,}",
    )
    col4.metric(
        "Heuristic Confidence",
        f"{int(round(qee.confidence * 100))}%",
    )

    if qee.recommended_active_space is not None:
        st.info(
            f"**Recommended Reduced Active Space (HYBRID):** "
            f"`({qee.recommended_active_space[0]}e, {qee.recommended_active_space[1]}o)`"
        )

    st.markdown("**Routing Reasons:**")
    for reason in qee.reasons:
        st.markdown(f"- {reason}")

    if qee.warnings:
        st.markdown("**Methodological Warnings & Caveats:**")
        for warn in qee.warnings:
            st.warning(warn)


def render_history_tab(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Render the SQLite experiment history table and deterministic replay inspector."""
    st.subheader("SQLite Experiment History (`experiments` Table)")
    records = list_experiments(limit=200, db_path=db_path)

    if not records:
        st.info("No experiments stored in SQLite yet. Run an analysis in the Simulator tab first.")
        return

    summary_rows = []
    for r in records:
        summary_rows.append(
            {
                "ID": r["id"],
                "Timestamp (UTC)": r["timestamp"][:19].replace("T", " "),
                "Input": r["input_identifier"],
                "Run Type": r["config"].get("run_type", "single_seed_vqe"),
                "QEE Decision": r["qee_decision"],
                "Active Space": r["active_space"],
                "Noise Model": r["noise_model"] or "—",
                "Seed": r["seed"] if r["seed"] is not None else "—",
                "Qubits": r["qubits"],
                "Depth": r["depth"],
                "VQE Energy (Ha)": round(r["vqe_energy"], 6) if r["vqe_energy"] is not None else None,
                "FCI Energy (Ha)": round(r["fci_energy"], 6) if r["fci_energy"] is not None else None,
                "|Error| (mHa)": round(r["error_ha"] * 1000.0, 4) if r["error_ha"] is not None else None,
                "Reliability": r["reliability"] or "—",
                "Verdict": r["verdict"] or "—",
            }
        )

    df_hist = pd.DataFrame(summary_rows)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Deterministic Experiment Replay by ID")
    st.caption(
        "Select any stored experiment ID from SQLite to re-execute its exact saved configuration "
        "and verify whether it reproduces the same QEE decision and VQE energy."
    )
    available_ids = [int(r["id"]) for r in records]
    selected_id = st.selectbox("Select Experiment ID to Replay", options=available_ids)

    if st.button("Replay Stored Experiment", key="replay_btn"):
        with st.spinner(f"Replaying experiment #{selected_id} from saved config_json..."):
            rep = replay_experiment(selected_id, db_path=db_path)

        if rep.reproduced:
            st.success(
                f"Experiment #{rep.experiment_id} (`{rep.input_identifier}`, `{rep.run_type}`) "
                f"reproduced identically (`reproduced = True`)."
            )
        else:
            st.error(
                f"Experiment #{rep.experiment_id} showed a discrepancy (`reproduced = False`)."
            )

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Stored QEE Decision", rep.stored_qee_decision)
        rcol2.metric("Replayed QEE Decision", rep.replayed_qee_decision)
        rcol3.metric(
            "Stored VQE Energy",
            f"{rep.stored_vqe_energy:.8f} Ha" if rep.stored_vqe_energy is not None else "N/A",
        )
        rcol4.metric(
            "Replayed VQE Energy (|ΔE|)",
            f"{rep.replayed_vqe_energy:.8f} Ha" if rep.replayed_vqe_energy is not None else "N/A",
            delta=f"{rep.energy_diff_ha:.2e} Ha" if rep.energy_diff_ha is not None else None,
        )


def main() -> None:
    """Main Streamlit application entry point for Q-Med."""
    st.set_page_config(
        page_title="Q-Med | Quantum Chemistry Eligibility & VQE Benchmark",
        layout="wide",
    )

    st.title("Q-Med: Quantum Chemistry Eligibility & VQE Decision-Support Prototype")
    st.caption(
        "Research & educational prototype evaluating when small-molecule quantum simulation (VQE) "
        "is worth running against exact PySCF Hartree–Fock and FCI/CASCI baselines. "
        "Not a clinical or drug-approval tool."
    )

    with st.sidebar:
        st.header("1. Molecule Input")
        input_mode = st.radio(
            "Select Input Mode",
            options=["Preset Benchmark / Demo Molecule", "Custom SMILES String"],
        )

        if input_mode == "Preset Benchmark / Demo Molecule":
            preset_label = st.selectbox("Choose Molecule", options=list(PRESET_MOLECULES.keys()))
            target_input = PRESET_MOLECULES[preset_label]
        else:
            target_input = st.text_input(
                "Enter SMILES or Benchmark ID",
                value="O",
                help="Examples: [H][H] (H2), [LiH] (LiH), O (Water), CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
            )

        st.header("2. Simulation Settings")
        num_seeds = st.slider(
            "Number of Random Seeds (min 3 for verdict)",
            min_value=3,
            max_value=5,
            value=5,
        )
        seeds: Tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS[:num_seeds]

        run_noisy = st.checkbox("Run Noisy Qiskit Aer Simulation", value=True)
        noisy_mode = st.selectbox(
            "Noise Model",
            options=["depolarizing_readout", "fake_backend"],
            disabled=not run_noisy,
        )
        shots = st.select_slider(
            "Measurement Shots (Noisy Mode)",
            options=[256, 512, 1024, 2048],
            value=1024,
            disabled=not run_noisy,
        )
        ideal_maxiter = st.number_input(
            "Ideal SLSQP Max Iterations",
            min_value=5,
            max_value=200,
            value=40,
        )
        noisy_maxiter = st.number_input(
            "Noisy COBYLA Max Evaluations",
            min_value=4,
            max_value=100,
            value=20,
            disabled=not run_noisy,
        )

        run_clicked = st.button("Run Q-Med Pipeline", type="primary", use_container_width=True)

    tab_sim, tab_scan, tab_hist = st.tabs(
        [
            "1. Eligibility & VQE Benchmark",
            "2. H2 Potential Energy Surface & Water Demos",
            "3. SQLite History & Replay",
        ]
    )

    with tab_sim:
        resolved_smiles, bm_id = resolve_input_smiles(target_input)
        val_res = validate_and_describe(resolved_smiles)

        if not val_res.is_valid or val_res.descriptors is None:
            st.error(val_res.error or f"Invalid SMILES input: '{target_input}'")
        else:
            desc = val_res.descriptors
            st.subheader(f"Physicochemical Descriptors (RDKit) — `{desc.formula}`")
            d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
            d1.metric("MW (g/mol)", f"{desc.mw:.2f}")
            d2.metric("LogP", f"{desc.logp:.2f}")
            d3.metric("HBD", f"{desc.hbd}")
            d4.metric("HBA", f"{desc.hba}")
            d5.metric("TPSA (Å²)", f"{desc.tpsa:.2f}")
            d6.metric("Rot. Bonds", f"{desc.rotatable_bonds}")
            d7.metric("QED", f"{desc.qed:.3f}")

            qee_preview = evaluate_eligibility(target_input)
            st.divider()
            render_qee_card(qee_preview)

            if run_clicked:
                with st.spinner(
                    f"Running Q-Med pipeline for '{target_input}' across seeds {list(seeds)}..."
                ):
                    pipe_res = run_qmed_pipeline(
                        input_identifier=target_input,
                        seeds=seeds,
                        run_noisy=run_noisy,
                        noisy_mode=noisy_mode,
                        shots=int(shots),
                        ideal_maxiter=int(ideal_maxiter),
                        noisy_maxiter=int(noisy_maxiter),
                        db_path=DEFAULT_DB_PATH,
                    )
                st.session_state["last_pipeline_result"] = pipe_res

            pipe_res_display: Optional[PipelineExecutionResult] = st.session_state.get(
                "last_pipeline_result"
            )
            if (
                pipe_res_display is not None
                and pipe_res_display.input_identifier == target_input
            ):
                st.divider()
                st.subheader("Benchmark Results: VQE vs. PySCF Hartree–Fock & Exact FCI/CASCI")

                if pipe_res_display.ideal_benchmark is None:
                    st.info(
                        f"Molecule routed to **{pipe_res_display.qee.decision}**. "
                        f"Quantum circuit execution was skipped per QEE v1 rules, and record "
                        f"`#{pipe_res_display.primary_experiment_id}` was saved to SQLite."
                    )
                else:
                    ideal_bm = pipe_res_display.ideal_benchmark
                    b1, b2, b3, b4 = st.columns(4)
                    b1.markdown(
                        f"**Ideal Reliability**<br>{format_badge_html(ideal_bm.reliability, RELIABILITY_COLORS)}",
                        unsafe_allow_html=True,
                    )
                    b2.markdown(
                        f"**Ideal Verdict**<br>`{ideal_bm.verdict}`",
                        unsafe_allow_html=True,
                    )
                    if pipe_res_display.noisy_benchmark is not None:
                        noisy_bm = pipe_res_display.noisy_benchmark
                        b3.markdown(
                            f"**Noisy Reliability**<br>{format_badge_html(noisy_bm.reliability, RELIABILITY_COLORS)}",
                            unsafe_allow_html=True,
                        )
                        b4.markdown(
                            f"**Noisy Verdict**<br>`{noisy_bm.verdict}`",
                            unsafe_allow_html=True,
                        )

                    df_res = build_results_dataframe(pipe_res_display)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                    all_warnings = list(
                        dict.fromkeys(
                            ideal_bm.warnings
                            + (
                                pipe_res_display.noisy_benchmark.warnings
                                if pipe_res_display.noisy_benchmark
                                else []
                            )
                        )
                    )
                    if all_warnings:
                        st.markdown("**Physical & Numerical Diagnostics Flagged:**")
                        for w in all_warnings:
                            st.warning(w)

                    fig = build_ideal_vs_noisy_figure(pipe_res_display)
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(
                        f"Saved {len(pipe_res_display.stored_experiment_ids)} experiment records "
                        f"(IDs: `{pipe_res_display.stored_experiment_ids}`) to `{DEFAULT_DB_PATH.name}`."
                    )

    with tab_scan:
        st.subheader("H2 Bond-Length Dissociation Curve (0.30 Å – 2.50 Å)")
        curve_path = PROJECT_ROOT / "docs" / "h2_curve.png"
        if curve_path.exists():
            st.image(
                str(curve_path),
                caption="Warm-started UCCSD VQE vs. Exact PySCF FCI and RHF across H2 bond lengths (STO-3G).",
                use_container_width=True,
            )
        else:
            st.warning("docs/h2_curve.png not found. Run notebooks/h2_curve.ipynb to generate.")

        st.divider()
        st.subheader("Water Option B: (8e, 6o) EfficientSU2(reps=1) Ansatz-Limitation Demonstration")
        st.caption(
            "Demonstrates why a shallow hardware-efficient ansatz (EfficientSU2, reps=1, 40 parameters, "
            "depth 11 on 10 qubits with Parity mapping) cannot recover the full correlation energy of "
            "H2O (8e, 6o), leaving ~29–46 mHa residual error vs. CASCI(8e, 6o)."
        )
        if st.button("Run Water (8e, 6o) EfficientSU2 Ansatz-Limitation Demo (3 Seeds)"):
            with st.spinner("Running 3-seed EfficientSU2(reps=1) demo on H2O (8e, 6o)..."):
                demo_runs = run_water_option_b_demo(seeds=(42, 43, 44), maxiter=80)
            demo_rows = [
                {
                    "Seed": r.seed,
                    "Active Space": f"({r.active_electrons}e, {r.active_orbitals}o)",
                    "Mapping": r.mapping,
                    "Qubits": r.num_qubits,
                    "Ansatz": f"{r.ansatz}(reps=1)",
                    "Depth / Gates": f"{r.circuit_depth} / {r.gate_count}",
                    "VQE Energy (Ha)": round(r.vqe_energy, 6),
                    "Matched CASCI(8e,6o) (Ha)": round(-75.01232538, 6),
                    "Error vs CASCI (mHa)": round((r.vqe_energy - (-75.01232538)) * 1000.0, 2),
                    "Iterations": r.iterations,
                    "Reliability": r.reliability,
                    "Label": r.run_label,
                }
                for r in demo_runs
            ]
            st.dataframe(pd.DataFrame(demo_rows), use_container_width=True, hide_index=True)

    with tab_hist:
        render_history_tab(db_path=DEFAULT_DB_PATH)


if __name__ == "__main__":
    main()
