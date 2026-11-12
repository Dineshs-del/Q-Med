"""Rule-based Quantum Eligibility Engine (QEE v1)."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from qmed.active_space import estimate_circuit_depth, estimate_qubits_from_smiles
from qmed.config import (
    DEFAULT_HYBRID_ACTIVE_SPACE,
    MAX_CIRCUIT_DEPTH,
    MAX_HYBRID_FULL_QUBITS,
    MAX_SIMULATED_QUBITS,
    MIN_ACTIVE_ELECTRONS_FOR_QUANTUM,
    SUPPORTED_ELEMENTS,
)
from qmed.molecules import load_benchmarks, validate_and_describe

# Convenience aliases for common molecule names so users/tests can pass either name or SMILES
COMMON_NAME_TO_SMILES: Dict[str, str] = {
    "H2": "[H][H]",
    "HYDROGEN": "[H][H]",
    "LIH": "[LiH]",
    "LITHIUM HYDRIDE": "[LiH]",
    "H2O": "O",
    "WATER": "O",
    "ASPIRIN": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ETHANOL": "CCO",
    "HELIUM": "[He]",
    "NEON": "[Ne]",
    "AMMONIA": "N",
    "METHANE": "C",
}


@dataclass(frozen=True)
class QEEResult:
    """Output of the rule-based Quantum Eligibility Engine (QEE v1)."""

    is_valid: bool
    decision: Optional[str]
    confidence: float
    reasons: List[str]
    warnings: List[str]
    features: Dict[str, Any]
    recommended_active_space: Optional[Tuple[int, int]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the QEE routing decision."""
        return asdict(self)


def resolve_input_smiles(input_str: str) -> Tuple[str, Optional[str]]:
    """Resolve a benchmark ID, common name, or raw SMILES into (smiles, benchmark_id)."""
    if not isinstance(input_str, str) or not input_str.strip():
        return input_str, None

    cleaned = input_str.strip()
    upper = cleaned.upper()

    benchmarks = load_benchmarks()
    for bm_key, bm in benchmarks.items():
        if upper in {bm_key.upper(), bm.name.upper()} or cleaned == bm.smiles:
            return bm.smiles, bm.id

    if upper in COMMON_NAME_TO_SMILES:
        resolved = COMMON_NAME_TO_SMILES[upper]
        for bm in benchmarks.values():
            if resolved == bm.smiles:
                return resolved, bm.id
        return resolved, None

    return cleaned, None


def evaluate_eligibility(
    smiles: str,
    estimated_qubits: Optional[int] = None,
    estimated_depth: Optional[int] = None,
    reference_available: bool = True,
    reducible_to_limit: Optional[bool] = None,
    recommended_active_space: Optional[Tuple[int, int]] = None,
    classical_sufficient: bool = False,
) -> QEEResult:
    """Route a molecule to CLASSICAL_ONLY, SIMULATED_QUANTUM, HYBRID, or DEFERRED."""
    resolved_smiles, benchmark_id = resolve_input_smiles(smiles)
    val = validate_and_describe(resolved_smiles)
    if not val.is_valid or val.descriptors is None:
        return QEEResult(
            is_valid=False,
            decision=None,
            confidence=0.0,
            reasons=[],
            warnings=[],
            features={"input_smiles": smiles},
            error=val.error or "Invalid SMILES string.",
        )

    desc = val.descriptors
    unsupported = [el for el in desc.elements if el not in SUPPORTED_ELEMENTS]
    if unsupported:
        err_msg = (
            f"Unsupported element(s) for STO-3G quantum workflow: {', '.join(unsupported)}. "
            f"Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}."
        )
        return QEEResult(
            is_valid=False,
            decision=None,
            confidence=0.0,
            reasons=[],
            warnings=[],
            features={"input_smiles": smiles, "elements": desc.elements},
            error=err_msg,
        )

    mapping_hint = "parity" if benchmark_id in {"H2", "LiH"} else None
    est_summary = estimate_qubits_from_smiles(
        desc.canonical_smiles if benchmark_id is None else resolved_smiles,
        frozen_core=True,
        mapping=mapping_hint,
    )
    qubits = int(estimated_qubits) if estimated_qubits is not None else est_summary.num_qubits

    if estimated_depth is not None:
        depth = int(estimated_depth)
    else:
        depth = int(est_summary.estimated_circuit_depth)

    # Determine whether the molecule can be meaningfully reduced to a frontier active space
    if reducible_to_limit is None:
        can_reduce = qubits <= MAX_HYBRID_FULL_QUBITS
    else:
        can_reduce = bool(reducible_to_limit)

    features: Dict[str, Any] = {
        "input_identifier": smiles,
        "canonical_smiles": desc.canonical_smiles,
        "benchmark_id": benchmark_id,
        "formula": desc.formula,
        "mw": desc.mw,
        "elements": desc.elements,
        "total_electrons": est_summary.total_electrons,
        "total_spatial_orbitals": est_summary.total_spatial_orbitals,
        "frozen_core_orbitals": est_summary.frozen_core_orbitals,
        "active_electrons": est_summary.active_electrons,
        "active_spatial_orbitals": est_summary.active_spatial_orbitals,
        "mapping": est_summary.mapping,
        "estimated_qubits": qubits,
        "estimated_uccsd_parameters": est_summary.estimated_uccsd_parameters,
        "estimated_circuit_depth": depth,
        "max_simulated_qubits": MAX_SIMULATED_QUBITS,
        "max_circuit_depth": MAX_CIRCUIT_DEPTH,
        "max_hybrid_full_qubits": MAX_HYBRID_FULL_QUBITS,
        "reference_available": reference_available,
    }

    reasons: List[str] = []
    warnings: List[str] = [
        "Confidence score is a rule-based heuristic reflecting distance from resource thresholds."
    ]

    # 1. CLASSICAL_ONLY: closed-shell / zero-virtual-orbital systems or explicit classical sufficiency
    if (
        classical_sufficient
        or est_summary.active_electrons < MIN_ACTIVE_ELECTRONS_FOR_QUANTUM
        or est_summary.estimated_uccsd_parameters == 0
    ):
        if classical_sufficient:
            reasons.append(
                "Classical exact solver (Hartree-Fock / FCI) marked sufficient for this system; "
                "skipping quantum circuit execution."
            )
        elif est_summary.active_electrons < MIN_ACTIVE_ELECTRONS_FOR_QUANTUM:
            reasons.append(
                f"Active space contains only {est_summary.active_electrons} valence electron(s) "
                f"(< {MIN_ACTIVE_ELECTRONS_FOR_QUANTUM}); electron-pair correlation is absent, "
                "so classical Hartree-Fock / exact diagonalization is exact."
            )
        else:
            reasons.append(
                f"In STO-3G frozen-core ({est_summary.active_electrons}e, "
                f"{est_summary.active_spatial_orbitals}o), all spatial orbitals are doubly occupied "
                "(0 virtual orbitals, 0 UCCSD excitation parameters); classical Hartree-Fock is exact."
            )
        warnings.append(
            "No quantum circuit excitations are required because correlation energy in this basis is zero."
        )
        return QEEResult(
            is_valid=True,
            decision="CLASSICAL_ONLY",
            confidence=0.95,
            reasons=reasons,
            warnings=warnings,
            features=features,
        )

    # 2. DEFERRED if classical FCI/CASCI reference is explicitly unavailable
    if not reference_available:
        reasons.append(
            "Classical FCI/CASCI reference is unavailable for validation; deferring quantum simulation."
        )
        return QEEResult(
            is_valid=True,
            decision="DEFERRED",
            confidence=0.85,
            reasons=reasons,
            warnings=warnings,
            features=features,
        )

    # 3. SIMULATED_QUANTUM: fits within both MAX_SIMULATED_QUBITS and MAX_CIRCUIT_DEPTH
    if qubits <= MAX_SIMULATED_QUBITS and depth <= MAX_CIRCUIT_DEPTH:
        reasons.append(
            f"Estimated qubits ({qubits} <= {MAX_SIMULATED_QUBITS}) and UCCSD circuit depth "
            f"({depth} <= {MAX_CIRCUIT_DEPTH}) are within full active-space simulation limits."
        )
        if est_summary.active_electrons <= 2:
            warnings.append(
                "2-active-electron system: UCCSD is exact in this subspace and serves as pipeline validation."
            )
        return QEEResult(
            is_valid=True,
            decision="SIMULATED_QUANTUM",
            confidence=0.90,
            reasons=reasons,
            warnings=warnings,
            features=features,
        )

    # 4. HYBRID: either depth exceeds MAX_CIRCUIT_DEPTH at <=12 qubits (e.g. water (8e,6o)),
    #    or qubits moderately exceed 12 (<= MAX_HYBRID_FULL_QUBITS) and can be reduced to (4e,4o).
    if (qubits <= MAX_SIMULATED_QUBITS and depth > MAX_CIRCUIT_DEPTH) or (
        qubits > MAX_SIMULATED_QUBITS and can_reduce
    ):
        rec_space = recommended_active_space or DEFAULT_HYBRID_ACTIVE_SPACE
        red_qubits = 2 * rec_space[1]
        _, red_depth = estimate_circuit_depth(rec_space[0], rec_space[1], red_qubits, "UCCSD")

        if depth > MAX_CIRCUIT_DEPTH and qubits <= MAX_SIMULATED_QUBITS:
            reasons.append(
                f"Full frozen-core active space ({est_summary.active_electrons}e, "
                f"{est_summary.active_spatial_orbitals}o) fits in {qubits} qubits (<={MAX_SIMULATED_QUBITS}), "
                f"but estimated UCCSD circuit depth ({depth}) exceeds MAX_CIRCUIT_DEPTH ({MAX_CIRCUIT_DEPTH})."
            )
            reasons.append(
                f"Routing to HYBRID: reducing active space to ({rec_space[0]}e, {rec_space[1]}o) "
                f"({red_qubits} qubits, depth ~{red_depth} <= {MAX_CIRCUIT_DEPTH}) so VQE circuit depth stays within limit."
            )
        else:
            reasons.append(
                f"Full frozen-core qubit count ({qubits}) exceeds MAX_SIMULATED_QUBITS ({MAX_SIMULATED_QUBITS}) "
                f"but is within MAX_HYBRID_FULL_QUBITS ({MAX_HYBRID_FULL_QUBITS}); reducing active space to "
                f"({rec_space[0]}e, {rec_space[1]}o) ({red_qubits} qubits, depth ~{red_depth})."
            )

        warnings.append(
            f"Active-space reduction to {rec_space} omits outer-valence correlation; "
            "active_space_error_mHa will be tracked and caps maximum reliability at MEDIUM."
        )
        return QEEResult(
            is_valid=True,
            decision="HYBRID",
            confidence=0.80,
            reasons=reasons,
            warnings=warnings,
            features=features,
            recommended_active_space=rec_space,
        )

    # 5. DEFERRED: large molecules (such as aspirin, 120 frozen-core qubits) exceeding simulator & hybrid limits
    reasons.append(
        f"Full frozen-core STO-3G active space ({est_summary.active_electrons}e, "
        f"{est_summary.active_spatial_orbitals}o) requires {qubits} qubits "
        f"(exceeds MAX_SIMULATED_QUBITS={MAX_SIMULATED_QUBITS} and MAX_HYBRID_FULL_QUBITS={MAX_HYBRID_FULL_QUBITS}) "
        f"and an estimated UCCSD circuit depth of {depth:,} (> MAX_CIRCUIT_DEPTH={MAX_CIRCUIT_DEPTH})."
    )
    reasons.append(
        f"Truncating {est_summary.active_electrons} valence electrons down to a <={MAX_SIMULATED_QUBITS}-qubit "
        "active space would discard the vast majority of valence correlation, and exact full-valence classical "
        "FCI validation is intractable at this scale."
    )
    warnings.append(
        "Molecule is too large for reliable NISQ/statevector VQE simulation without severe active-space truncation error."
    )
    return QEEResult(
        is_valid=True,
        decision="DEFERRED",
        confidence=0.90,
        reasons=reasons,
        warnings=warnings,
        features=features,
    )
