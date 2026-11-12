"""Streamlit decision-support UI for Q-Med: Interactive 3D/2D Viewer, Plain-Language Help, QEE, VQE vs FCI, and SQLite History."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st
import streamlit.components.v1 as components

from qmed.config import (
    DEFAULT_BENCHMARK_SEEDS,
    DEFAULT_DB_PATH,
    HIGH_RELIABILITY_ERROR_HA,
    MAX_CIRCUIT_DEPTH,
    MAX_SIMULATED_QUBITS,
    PROJECT_ROOT,
)
from qmed.molecules import (
    APPROXIMATE_GEOMETRY_CAPTION,
    CPK_ELEMENT_COLORS,
    MoleculeValidationError,
    Structure3DData,
    load_molecule_3d_structure,
    validate_and_describe,
)
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

BENCHMARK_PRESET_IDS = {"H2", "LiH", "H2O"}

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

PLAIN_LANGUAGE_GLOSSARY: Dict[str, str] = {
    "SMILES": (
        "Simplified Molecular Input Line Entry System — a compact text notation for writing "
        "chemical structures using ASCII characters (for example, 'O' for water, 'CCO' for ethanol, "
        "or 'CC(=O)OC1=CC=CC=C1C(=O)O' for aspirin)."
    ),
    "Hartree": (
        "The standard atomic unit of energy in quantum chemistry (1 Hartree ≈ 627.5 kcal/mol ≈ 27.21 eV). "
        "Bound electronic ground-state energies are negative because bound electrons have lower energy than separated particles."
    ),
    "mHa": (
        "Milli-Hartree (0.001 Hartree ≈ 0.6275 kcal/mol). 'Chemical accuracy' — the accuracy needed to "
        "predict room-temperature chemical reaction rates — is 1 kcal/mol, or about 1.6 mHa."
    ),
    "HF": (
        "Hartree–Fock — a fast classical baseline method that approximates each electron as moving in the "
        "average electric field of all other electrons. It misses instantaneous electron-electron avoidance (correlation energy)."
    ),
    "FCI/CASCI": (
        "Full Configuration Interaction (FCI) / Complete Active Space CI (CASCI) — the exact mathematical "
        "ground-state energy within the chosen basis set and active orbital window, computed classically with PySCF. "
        "Because FCI cost grows exponentially with system size, it is only tractable for small active spaces."
    ),
    "VQE": (
        "Variational Quantum Eigensolver — a hybrid quantum-classical algorithm where a quantum circuit "
        "prepares a trial molecular state and a classical optimizer adjusts the circuit angles to minimize the measured energy."
    ),
    "UCCSD": (
        "Unitary Coupled Cluster Singles and Doubles — a chemistry-inspired quantum circuit ansatz that "
        "excites one or two electrons at a time from occupied to virtual orbitals while preserving quantum unitarity."
    ),
    "qubit": (
        "The basic unit of quantum information. In molecular simulation, each spin-orbital (an orbital with "
        "spin-up or spin-down) typically maps to one qubit, minus two qubits when parity symmetry reduction is used."
    ),
    "seed": (
        "A fixed starting number for the random number generator. Running VQE across multiple seeds (e.g., 42..46) "
        "tests whether the optimizer reliably finds the same ground state from different starting guesses."
    ),
    "shots": (
        "The number of repeated quantum circuit measurements used to estimate an average energy in noisy or "
        "hardware simulation. More shots reduce statistical sampling noise (which scales as 1/√shots)."
    ),
    "noise": (
        "Physical hardware imperfections modeled in Qiskit Aer, including gate depolarizing errors (random qubit "
        "flips during 1-qubit and 2-qubit gates) and readout errors (misclassifying a 0 as 1 during measurement)."
    ),
    "active space": (
        "A selected subset of chemically important valence electrons and molecular orbitals — written as (Ne, No) — "
        "simulated on the quantum circuit while tightly bound inner-core orbitals are frozen classically."
    ),
    "reliability labels": (
        "Q-Med's trust rating for a VQE run: HIGH (error ≤ 1.6 mHa, seed spread ≤ 1.6 mHa, and active-space truncation "
        "≤ 1.6 mHa), MEDIUM (error ≤ 10 mHa, or capped because active-space truncation exceeds 1.6 mHa), LOW (error ≤ 50 mHa), "
        "UNRELIABLE (error > 50 mHa or unconverged), and INVALID (violates the variational lower bound on an ideal run)."
    ),
    "quantum advantage": (
        "Whether the quantum simulation outperformed classical exact methods in accuracy and speed. For small molecules "
        "where classical PySCF FCI solves the exact energy in milliseconds, the honest verdict is NO_DEMONSTRATED_ADVANTAGE."
    ),
}


def format_badge_html(label: str, color_map: Dict[str, str], tooltip: str = "") -> str:
    """Render an inline HTML badge with an optional hover title attribute."""
    bg = color_map.get(label, "#475569")
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<span{title_attr} style="background-color:{bg};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:600;font-size:0.85rem;cursor:help;">{label}</span>'
    )


def render_rdkit_2d_image(smiles: str, size: Tuple[int, int] = (280, 240)) -> Optional[Any]:
    """Render a 2D chemical structure depiction (PIL Image) from a SMILES string using RDKit Draw."""
    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def build_py3dmol_html(
    struct: Structure3DData,
    style: str = "Ball-and-stick",
    show_labels: bool = True,
    show_hydrogens: bool = True,
    spin: bool = False,
    background: str = "Light",
    width: int = 480,
    height: int = 340,
) -> str:
    """Build self-contained HTML for py3Dmol using view._make_html() without stmol."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(struct.xyz_block, "xyz")

    style_norm = style.strip().lower()
    if style_norm in {"ball-and-stick", "ball_and_stick"}:
        style_spec: Dict[str, Any] = {
            "stick": {"radius": 0.14, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.28, "colorscheme": "Jmol"},
        }
    elif style_norm == "stick":
        style_spec = {"stick": {"radius": 0.20, "colorscheme": "Jmol"}}
    elif style_norm in {"space-filling", "space_filling", "vdw"}:
        style_spec = {"sphere": {"scale": 0.85, "colorscheme": "Jmol"}}
    elif style_norm == "wireframe":
        style_spec = {"line": {"linewidth": 3.0, "colorscheme": "Jmol"}}
    else:
        style_spec = {
            "stick": {"radius": 0.14, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.28, "colorscheme": "Jmol"},
        }

    view.setStyle({}, style_spec)

    has_non_h = any(a.symbol != "H" for a in struct.atoms)
    if not show_hydrogens and has_non_h:
        view.setStyle({"elem": "H"}, {})

    bg_hex = "#0f172a" if background.strip().lower() == "dark" else "#f8fafc"
    view.setBackgroundColor(bg_hex)

    if show_labels:
        lbl_bg = "#1e293b" if background.strip().lower() == "dark" else "#ffffff"
        lbl_fg = "#f8fafc" if background.strip().lower() == "dark" else "#0f172a"
        for idx, atom in enumerate(struct.atoms):
            if not show_hydrogens and has_non_h and atom.symbol == "H":
                continue
            view.addLabel(
                f"{atom.symbol}{idx + 1}",
                {
                    "position": {
                        "x": float(atom.coords[0]),
                        "y": float(atom.coords[1]),
                        "z": float(atom.coords[2]),
                    },
                    "backgroundColor": lbl_bg,
                    "fontColor": lbl_fg,
                    "fontSize": 12,
                    "backgroundOpacity": 0.75,
                },
            )

    view.zoomTo()
    view.spin(bool(spin))
    return str(view._make_html())


def build_plotly_3d_molecule_figure(
    struct: Structure3DData,
    show_labels: bool = True,
    show_hydrogens: bool = True,
    background: str = "Light",
) -> go.Figure:
    """Fallback 3D ball-and-stick Plotly figure if py3Dmol fails."""
    fig = go.Figure()
    has_non_h = any(a.symbol != "H" for a in struct.atoms)

    # Draw covalent bond segments
    for i, j in struct.bonds:
        ai, aj = struct.atoms[i], struct.atoms[j]
        if not show_hydrogens and has_non_h and (ai.symbol == "H" or aj.symbol == "H"):
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[ai.coords[0], aj.coords[0]],
                y=[ai.coords[1], aj.coords[1]],
                z=[ai.coords[2], aj.coords[2]],
                mode="lines",
                line={"color": "#94a3b8", "width": 6},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw atoms with CPK colors
    xs, ys, zs, colors, sizes, labels = [], [], [], [], [], []
    for idx, atom in enumerate(struct.atoms):
        if not show_hydrogens and has_non_h and atom.symbol == "H":
            continue
        xs.append(atom.coords[0])
        ys.append(atom.coords[1])
        zs.append(atom.coords[2])
        colors.append(CPK_ELEMENT_COLORS.get(atom.symbol, "#38bdf8"))
        sizes.append(10 if atom.symbol == "H" else 16)
        labels.append(f"{atom.symbol}{idx + 1}")

    mode = "markers+text" if show_labels else "markers"
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode=mode,
            text=labels if show_labels else None,
            textposition="top center",
            marker={
                "size": sizes,
                "color": colors,
                "line": {"color": "#0f172a", "width": 1},
            },
            showlegend=False,
        )
    )

    bg_color = "#0f172a" if background.strip().lower() == "dark" else "#f8fafc"
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=340,
        paper_bgcolor=bg_color,
        scene={
            "bgcolor": bg_color,
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
    )
    return fig


def build_geometry_tables(struct: Structure3DData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build DataFrames of bond lengths (Å) and bond angles (°) from Structure3DData."""
    bond_rows = [
        {
            "Bond": b.bond_label,
            "Length (Å)": f"{b.distance_angstrom:.4f}",
        }
        for b in struct.bond_lengths
    ]
    angle_rows = [
        {
            "Angle": a.angle_label,
            "Degrees (°)": f"{a.angle_degrees:.2f}°",
        }
        for a in struct.bond_angles
    ]
    return pd.DataFrame(bond_rows), pd.DataFrame(angle_rows)


def build_results_dataframe(result: PipelineExecutionResult) -> pd.DataFrame:
    """Build a comparison DataFrame of Classical HF, Exact FCI/CASCI, Ideal VQE, and Noisy VQE."""
    rows: List[Dict[str, Any]] = []
    if result.ideal_benchmark is None:
        return pd.DataFrame(rows)

    ideal = result.ideal_benchmark
    space_str = f"({ideal.active_electrons}e, {ideal.active_orbitals}o)"

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


def build_error_vs_exact_figure(result: PipelineExecutionResult) -> go.Figure:
    """Create a Plotly bar chart of |Error vs. Exact FCI/CASCI| (mHa) with a 1.6 mHa chemical accuracy line."""
    fig = go.Figure()
    if result.ideal_benchmark is None:
        return fig

    ideal = result.ideal_benchmark
    hf_err_mha = abs(ideal.hf_energy - ideal.matched_fci_energy) * 1000.0

    methods = [
        "Classical Hartree–Fock (PySCF)",
        f"VQE Ideal ({ideal.ansatz} + {ideal.optimizer})",
    ]
    errors_mha = [hf_err_mha, ideal.abs_mean_error_mha]
    spreads_mha = [0.0, ideal.uncertainty_mha]
    colors = ["#64748b", "#16a34a"]

    if result.noisy_benchmark is not None:
        noisy = result.noisy_benchmark
        methods.append(f"VQE Noisy [{noisy.noise_mode}]")
        errors_mha.append(noisy.abs_mean_error_mha)
        spreads_mha.append(noisy.uncertainty_mha)
        colors.append("#dc2626")

    fig.add_trace(
        go.Bar(
            x=methods,
            y=errors_mha,
            error_y={"type": "data", "array": spreads_mha, "visible": True},
            marker_color=colors,
            text=[f"{err:.4f} ± {std:.4f} mHa" for err, std in zip(errors_mha, spreads_mha)],
            textposition="outside",
            name="|Error vs. Exact| (mHa)",
        )
    )

    chem_acc_mha = HIGH_RELIABILITY_ERROR_HA * 1000.0
    fig.add_hline(
        y=chem_acc_mha,
        line_dash="dash",
        line_color="#16a34a",
        line_width=2,
        annotation_text=f"Chemical Accuracy Threshold ({chem_acc_mha:.1f} mHa)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=(
            f"{ideal.molecule_id} ({ideal.active_electrons}e, {ideal.active_orbitals}o) — "
            "Absolute Error vs. Exact FCI/CASCI (mHa, Lower is Better)"
        ),
        yaxis_title="|Energy − Matched FCI/CASCI| (mHa)",
        xaxis_title="Method",
        template="plotly_white",
        height=440,
    )
    return fig


# Keep backward-compatible alias for existing tests
build_ideal_vs_noisy_figure = build_error_vs_exact_figure


def render_glossary_expander() -> None:
    """Render the plain-language 'What am I looking at?' glossary expander."""
    with st.expander("What am I looking at? (Plain-Language Glossary)"):
        st.markdown(
            "New to quantum chemistry? Here is a quick guide to every term displayed on this page:"
        )
        g_cols = st.columns(2)
        items = list(PLAIN_LANGUAGE_GLOSSARY.items())
        half = (len(items) + 1) // 2
        for col_idx, subset in enumerate([items[:half], items[half:]]):
            with g_cols[col_idx]:
                for term, definition in subset:
                    st.markdown(f"- **{term}**: {definition}")


def render_molecule_viewer_and_descriptors(
    target_input: str,
    is_preset_mode: bool,
) -> bool:
    """Render the interactive 3D/2D viewer beside RDKit descriptors at the top of Tab 1."""
    resolved_smiles, bm_id = resolve_input_smiles(target_input)
    val_res = validate_and_describe(resolved_smiles)

    if not val_res.is_valid or val_res.descriptors is None:
        st.warning(
            f"Could not parse '{target_input}' as a valid molecule. "
            f"{val_res.error or 'Please check your SMILES string (for example: O for water, CCO for ethanol).'}"
        )
        return False

    desc = val_res.descriptors
    use_bm_coords = bool(is_preset_mode and bm_id in BENCHMARK_PRESET_IDS)

    try:
        struct_3d = load_molecule_3d_structure(
            target_input if use_bm_coords else resolved_smiles,
            use_benchmark_coords=use_bm_coords,
            seed=42,
        )
    except MoleculeValidationError as exc:
        st.warning(f"Could not build 3D structure: {exc}")
        return False

    left_col, right_col = st.columns([1.45, 1.0], gap="large")

    with left_col:
        st.subheader(f"Interactive 3D & 2D Structure — `{desc.formula}`")
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.5, 1.0, 1.0, 1.0, 1.1])
        style_choice = ctrl1.selectbox(
            "3D Style",
            options=["Ball-and-stick", "Stick", "Space-filling", "Wireframe"],
            help="Choose how atoms and chemical bonds are drawn in the 3D viewer.",
        )
        show_labels = ctrl2.checkbox(
            "Atom labels",
            value=True,
            help="Show each atom's element symbol and index number (e.g., O1, H2) in 3D.",
        )
        show_hydrogens = ctrl3.checkbox(
            "Hydrogens",
            value=True,
            help="Show or hide hydrogen (H) atoms on heavy-atom molecules.",
        )
        auto_spin = ctrl4.checkbox(
            "Auto-spin",
            value=False,
            help="Continuously rotate the 3D molecule automatically.",
        )
        bg_choice = ctrl5.selectbox(
            "Background",
            options=["Light", "Dark"],
            help="Switch the 3D canvas between a light or dark background.",
        )

        v3d_col, v2d_col = st.columns([1.65, 1.0])
        with v3d_col:
            try:
                html_str = build_py3dmol_html(
                    struct_3d,
                    style=style_choice,
                    show_labels=show_labels,
                    show_hydrogens=show_hydrogens,
                    spin=auto_spin,
                    background=bg_choice,
                    width=430,
                    height=330,
                )
                components.html(html_str, height=345, scrolling=False)
            except Exception:
                fallback_fig = build_plotly_3d_molecule_figure(
                    struct_3d,
                    show_labels=show_labels,
                    show_hydrogens=show_hydrogens,
                    background=bg_choice,
                )
                st.plotly_chart(fallback_fig, use_container_width=True)

            st.caption(
                "Mouse: **Drag** to rotate • **Scroll** to zoom • **Right-drag** or **Shift+drag** to move."
            )
            st.caption(struct_3d.caption)

        with v2d_col:
            st.markdown("**2D Structure (RDKit)**")
            img_2d = render_rdkit_2d_image(desc.canonical_smiles, size=(240, 220))
            if img_2d is not None:
                st.image(img_2d, caption=f"SMILES: {desc.canonical_smiles}", use_container_width=True)
            else:
                st.caption(f"SMILES: `{desc.canonical_smiles}`")

    with right_col:
        st.subheader("Physicochemical Descriptors (RDKit)")
        m1, m2 = st.columns(2)
        m1.metric(
            "MW (g/mol)",
            f"{desc.mw:.2f}",
            help="Molecular Weight in grams per mole — total mass of all atoms in the molecule.",
        )
        m2.metric(
            "LogP",
            f"{desc.logp:.2f}",
            help="Octanol-water partition coefficient — measures how fat-loving (positive) or water-loving (negative) the molecule is.",
        )
        m3, m4 = st.columns(2)
        m3.metric(
            "HBD",
            f"{desc.hbd}",
            help="Hydrogen Bond Donors — number of N–H or O–H groups that can donate a hydrogen bond.",
        )
        m4.metric(
            "HBA",
            f"{desc.hba}",
            help="Hydrogen Bond Acceptors — number of nitrogen or oxygen atoms capable of accepting a hydrogen bond.",
        )
        m5, m6, m7 = st.columns(3)
        m5.metric(
            "TPSA (Å²)",
            f"{desc.tpsa:.2f}",
            help="Topological Polar Surface Area — surface area of polar atoms; values below 140 Å² generally favor cell membrane permeation.",
        )
        m6.metric(
            "Rot. Bonds",
            f"{desc.rotatable_bonds}",
            help="Rotatable Bonds — single non-ring bonds that allow the molecule to flex in 3D space.",
        )
        m7.metric(
            "QED",
            f"{desc.qed:.3f}",
            help="Quantitative Estimate of Drug-likeness (0 to 1) — combines molecular weight, polarity, and flexibility into an overall drug-likeness score.",
        )

        df_bonds, df_angles = build_geometry_tables(struct_3d)
        st.markdown("**Internal Geometry (from Displayed 3D Coordinates)**")
        if not df_bonds.empty:
            g1, g2 = st.columns([1.1, 1.0])
            with g1:
                st.caption("Bond Lengths (Å)")
                st.dataframe(df_bonds.head(10), use_container_width=True, hide_index=True, height=155)
            with g2:
                st.caption("Bond Angles (°)")
                if not df_angles.empty:
                    st.dataframe(df_angles.head(6), use_container_width=True, hide_index=True, height=155)
                else:
                    st.caption("Diatomic molecule (no 3-atom bond angle).")
        else:
            st.caption("Monoatomic species (no covalent bonds).")

    return True


def render_qee_card(qee: QEEResult) -> None:
    """Render the Quantum Eligibility Engine (QEE v1) decision card with plain-language tooltips."""
    st.subheader("Quantum Eligibility Engine (QEE v1) Decision")
    decision_label = qee.decision or "INVALID_INPUT"
    decision_tooltip = (
        "QEE routing outcome: SIMULATED_QUANTUM runs full frozen-core VQE; HYBRID reduces the active "
        "space to (4e, 4o) to keep gate depth manageable; CLASSICAL_ONLY skips VQE because classical "
        "Hartree-Fock is already exact; DEFERRED skips VQE because the molecule exceeds simulator limits."
    )
    badge_html = format_badge_html(decision_label, DECISION_COLORS, tooltip=decision_tooltip)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"**Routing Decision** ℹ️<br>{badge_html}",
        unsafe_allow_html=True,
        help=decision_tooltip,
    )
    col2.metric(
        "Estimated Qubits (Frozen-Core)",
        f"{qee.features.get('estimated_qubits', '—')} / {MAX_SIMULATED_QUBITS}",
        help="Number of quantum bits required to represent the active valence orbitals after freezing inner-shell core electrons (direct limit: 12 qubits).",
    )
    col3.metric(
        "Estimated UCCSD Depth",
        f"{qee.features.get('estimated_circuit_depth', 0):,} / {MAX_CIRCUIT_DEPTH:,}",
        help="Estimated number of sequential quantum gate layers in the decomposed UCCSD circuit (direct limit: 5,000 layers).",
    )
    col4.metric(
        "Heuristic Confidence",
        f"{int(round(qee.confidence * 100))}%",
        help="Rule-based score indicating how clearly the molecule falls inside or outside the qubit and circuit-depth thresholds.",
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
        st.info("No experiments stored in SQLite yet. Run an analysis in Tab 1 first.")
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
    selected_id = st.selectbox(
        "Select Experiment ID to Replay",
        options=available_ids,
        help="Choose a row ID from the SQLite experiments table to re-run from its saved configuration JSON.",
    )

    if st.button(
        "Replay Stored Experiment",
        key="replay_btn",
        help="Re-run the selected experiment with the exact same random seed and compare against the stored energy.",
    ):
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
        rcol1.metric(
            "Stored QEE Decision",
            rep.stored_qee_decision,
            help="The QEE routing decision originally saved in SQLite for this experiment ID.",
        )
        rcol2.metric(
            "Replayed QEE Decision",
            rep.replayed_qee_decision,
            help="The QEE routing decision produced when re-evaluating the saved configuration.",
        )
        rcol3.metric(
            "Stored VQE Energy",
            f"{rep.stored_vqe_energy:.8f} Ha" if rep.stored_vqe_energy is not None else "N/A",
            help="The VQE ground-state energy originally saved in SQLite.",
        )
        rcol4.metric(
            "Replayed VQE Energy (|ΔE|)",
            f"{rep.replayed_vqe_energy:.8f} Ha" if rep.replayed_vqe_energy is not None else "N/A",
            delta=f"{rep.energy_diff_ha:.2e} Ha" if rep.energy_diff_ha is not None else None,
            help="The VQE energy obtained on replay and its absolute difference from the stored value.",
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

    render_glossary_expander()

    with st.sidebar:
        st.header("1. Molecule Input")
        input_mode = st.radio(
            "Select Input Mode",
            options=["Preset Benchmark / Demo Molecule", "Custom SMILES String"],
            help="Choose a pre-configured benchmark molecule (with exact quantum reference coordinates) or type any custom chemical SMILES string.",
        )

        is_preset_mode = input_mode == "Preset Benchmark / Demo Molecule"
        if is_preset_mode:
            preset_label = st.selectbox(
                "Choose Molecule",
                options=list(PRESET_MOLECULES.keys()),
                help="Select a benchmark molecule (H2, LiH, H2O) or a demonstration molecule (Aspirin, Helium, Ammonia, Ethanol).",
            )
            target_input = PRESET_MOLECULES[preset_label]
        else:
            target_input = st.text_input(
                "Enter SMILES String",
                value="O",
                help="Type any valid SMILES text notation — for example: [H][H] (H2), O (water), CCO (ethanol), or CC(=O)OC1=CC=CC=C1C(=O)O (aspirin).",
            )

        st.header("2. Simulation Settings")
        num_seeds = st.slider(
            "Number of Random Seeds (min 3 for verdict)",
            min_value=3,
            max_value=5,
            value=5,
            help="Number of independent random starting guesses (seeds 42..46) used to measure optimizer consistency and uncertainty.",
        )
        seeds: Tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS[:num_seeds]

        run_noisy = st.checkbox(
            "Run Noisy Qiskit Aer Simulation",
            value=True,
            help="Also run VQE under simulated quantum hardware noise (gate depolarizing errors and measurement readout flips) to compare against the ideal noiseless run.",
        )
        noisy_mode = st.selectbox(
            "Noise Model",
            options=["depolarizing_readout", "fake_backend"],
            disabled=not run_noisy,
            help="Choose between a uniform 1Q/2Q depolarizing + readout bit-flip noise model or Qiskit's GenericBackendV2 calibrated fake device.",
        )
        shots = st.select_slider(
            "Measurement Shots (Noisy Mode)",
            options=[256, 512, 1024, 2048],
            value=1024,
            disabled=not run_noisy,
            help="Number of repeated circuit measurements per energy evaluation in noisy mode. Higher shots lower statistical sampling jitter.",
        )
        ideal_maxiter = st.number_input(
            "Ideal SLSQP Max Iterations",
            min_value=5,
            max_value=200,
            value=40,
            help="Maximum number of gradient-based SLSQP optimization steps for the ideal noiseless statevector VQE run.",
        )
        noisy_maxiter = st.number_input(
            "Noisy COBYLA Max Evaluations",
            min_value=4,
            max_value=100,
            value=20,
            disabled=not run_noisy,
            help="Maximum number of derivative-free COBYLA energy evaluations for the noisy VQE run.",
        )

        run_clicked = st.button(
            "Run Q-Med Pipeline",
            type="primary",
            use_container_width=True,
            help="Run the full Q-Med workflow: QEE eligibility routing, PySCF HF & FCI baselines, multi-seed VQE (if eligible), and SQLite logging.",
        )

    tab_sim, tab_scan, tab_hist = st.tabs(
        [
            "1. 3D Structure, Eligibility & VQE Benchmark",
            "2. H2 Potential Energy Surface & Water Demos",
            "3. SQLite History & Replay",
        ]
    )

    with tab_sim:
        valid_molecule = render_molecule_viewer_and_descriptors(
            target_input=target_input,
            is_preset_mode=is_preset_mode,
        )

        if valid_molecule:
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
                    rel_tooltip = (
                        "Reliability rating: HIGH (error ≤ 1.6 mHa, seed std ≤ 1.6 mHa, AS error ≤ 1.6 mHa), "
                        "MEDIUM (error ≤ 10 mHa or capped by active-space truncation), LOW (error ≤ 50 mHa), "
                        "UNRELIABLE (> 50 mHa or unconverged), INVALID (variational bound violation)."
                    )
                    verdict_tooltip = (
                        "Quantum-advantage verdict comparing VQE accuracy and runtime against exact PySCF FCI/CASCI."
                    )

                    b1, b2, b3, b4 = st.columns(4)
                    b1.markdown(
                        f"**Ideal Reliability** ℹ️<br>{format_badge_html(ideal_bm.reliability, RELIABILITY_COLORS, rel_tooltip)}",
                        unsafe_allow_html=True,
                        help=rel_tooltip,
                    )
                    b2.markdown(
                        f"**Ideal Verdict** ℹ️<br>`{ideal_bm.verdict}`",
                        unsafe_allow_html=True,
                        help=verdict_tooltip,
                    )
                    if pipe_res_display.noisy_benchmark is not None:
                        noisy_bm = pipe_res_display.noisy_benchmark
                        b3.markdown(
                            f"**Noisy Reliability** ℹ️<br>{format_badge_html(noisy_bm.reliability, RELIABILITY_COLORS, rel_tooltip)}",
                            unsafe_allow_html=True,
                            help=rel_tooltip,
                        )
                        b4.markdown(
                            f"**Noisy Verdict** ℹ️<br>`{noisy_bm.verdict}`",
                            unsafe_allow_html=True,
                            help=verdict_tooltip,
                        )

                    # 1. Error vs. Exact (mHa) Bar Chart with 1.6 mHa Chemical Accuracy Line
                    err_fig = build_error_vs_exact_figure(pipe_res_display)
                    st.plotly_chart(err_fig, use_container_width=True)

                    # 2. Flagged Warnings (including NOISE_ABOVE_HF_WARNING and RELIABILITY_CAPPED)
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

                    # 3. Full Total-Energy & Error Comparison Table below the chart
                    st.markdown("**Total Energies & Multi-Seed Benchmark Table**")
                    df_res = build_results_dataframe(pipe_res_display)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

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
        if st.button(
            "Run Water (8e, 6o) EfficientSU2 Ansatz-Limitation Demo (3 Seeds)",
            help="Run a 3-seed hardware-efficient VQE demo on water's 10-qubit valence space to illustrate ansatz expressibility limits.",
        ):
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
