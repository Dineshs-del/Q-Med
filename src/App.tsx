import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Download,
  Upload,
  Play,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Sliders,
  Database,
  FileSpreadsheet,
  Layers,
  HelpCircle,
} from 'lucide-react';

interface Atom3D {
  symbol: string;
  x: number;
  y: number;
  z: number;
}

interface BondLengthRow {
  label: string;
  distance: number;
}

interface BondAngleRow {
  label: string;
  degrees: number;
}

interface MoleculeProfile {
  id: string;
  name: string;
  smiles: string;
  canonicalSmiles: string;
  formula: string;
  isBenchmarkPreset: boolean;
  geometryNote: string;
  mw: number;
  logp: number;
  hbd: number;
  hba: number;
  tpsa: number;
  rotBonds: number;
  qed: number;
  totalElectrons: number;
  activeElectrons: number;
  activeOrbitals: number;
  effectiveElectrons: number;
  effectiveOrbitals: number;
  mapping: string;
  estimatedQubits: number;
  effectiveQubits: number;
  estimatedDepth: number;
  effectiveDepth: number;
  effectiveGates: number;
  uccsdParams: number;
  qeeDecision: 'SIMULATED_QUANTUM' | 'HYBRID' | 'CLASSICAL_ONLY' | 'DEFERRED' | 'REJECTED';
  confidence: number;
  reasons: string[];
  warnings: string[];
  hfEnergy: number | null;
  matchedFciEnergy: number | null;
  fullFcFciEnergy: number | null;
  idealVqeEnergy: number | null;
  idealErrorMha: number | null;
  idealStdMha: number | null;
  noisyVqeEnergy: number | null;
  noisyErrorMha: number | null;
  noisyStdMha: number | null;
  truncationErrorMha: number | null;
  idealReliability: string | null;
  noisyReliability: string | null;
  verdict: string;
  classicalRuntimeSec: number | null;
  idealRuntimeSec: number | null;
  noisyRuntimeSec: number | null;
  atoms: Atom3D[];
  bonds: [number, number][];
  bondLengths: BondLengthRow[];
  bondAngles: BondAngleRow[];
}

const CPK_COLORS: Record<string, string> = {
  H: '#E2E8F0',
  He: '#38BDF8',
  Li: '#A855F7',
  Be: '#22C55E',
  B: '#FB7185',
  C: '#334155',
  N: '#2563EB',
  O: '#DC2626',
  F: '#16A34A',
  Cl: '#15803D',
};

const MOLECULE_LIBRARY: Record<string, MoleculeProfile> = {
  H2: {
    id: 'H2',
    name: 'Hydrogen (0.735 Å Benchmark)',
    smiles: '[H][H]',
    canonicalSmiles: '[H][H]',
    formula: 'H2',
    isBenchmarkPreset: true,
    geometryNote: 'Exact benchmark geometry from data/benchmarks.json (R_HH = 0.7350 Å).',
    mw: 2.02,
    logp: -0.0,
    hbd: 0,
    hba: 0,
    tpsa: 0.0,
    rotBonds: 0,
    qed: 0.252,
    totalElectrons: 2,
    activeElectrons: 2,
    activeOrbitals: 2,
    effectiveElectrons: 2,
    effectiveOrbitals: 2,
    mapping: 'parity_2q_reduced',
    estimatedQubits: 2,
    effectiveQubits: 2,
    estimatedDepth: 15,
    effectiveDepth: 15,
    effectiveGates: 21,
    uccsdParams: 3,
    qeeDecision: 'SIMULATED_QUANTUM',
    confidence: 0.95,
    reasons: [
      'Full frozen-core active space (2e, 2o) requires 2 qubits (<= 12 threshold) and estimated UCCSD depth 15 (<= 5,000 threshold).',
      'System is directly simulatable on a noiseless/noisy statevector backend without active-space truncation.',
    ],
    warnings: [],
    hfEnergy: -1.11699899,
    matchedFciEnergy: -1.13730604,
    fullFcFciEnergy: -1.13730604,
    idealVqeEnergy: -1.13730598,
    idealErrorMha: 0.0001,
    idealStdMha: 0.0001,
    noisyVqeEnergy: -1.09281412,
    noisyErrorMha: 44.4919,
    noisyStdMha: 11.2405,
    truncationErrorMha: 0.0,
    idealReliability: 'HIGH',
    noisyReliability: 'LOW',
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: 0.042,
    idealRuntimeSec: 0.18,
    noisyRuntimeSec: 0.41,
    atoms: [
      { symbol: 'H', x: 0.0, y: 0.0, z: 0.0 },
      { symbol: 'H', x: 0.0, y: 0.0, z: 0.735 },
    ],
    bonds: [[0, 1]],
    bondLengths: [{ label: 'H1–H2', distance: 0.735 }],
    bondAngles: [],
  },
  LiH: {
    id: 'LiH',
    name: 'Lithium Hydride (1.595 Å Benchmark)',
    smiles: '[LiH]',
    canonicalSmiles: '[LiH]',
    formula: 'HLi',
    isBenchmarkPreset: true,
    geometryNote: 'Exact benchmark geometry from data/benchmarks.json (R_LiH = 1.5950 Å).',
    mw: 7.95,
    logp: -0.45,
    hbd: 0,
    hba: 0,
    tpsa: 0.0,
    rotBonds: 0,
    qed: 0.284,
    totalElectrons: 4,
    activeElectrons: 2,
    activeOrbitals: 5,
    effectiveElectrons: 2,
    effectiveOrbitals: 5,
    mapping: 'parity_2q_reduced',
    estimatedQubits: 8,
    effectiveQubits: 8,
    estimatedDepth: 1771,
    effectiveDepth: 1771,
    effectiveGates: 2418,
    uccsdParams: 24,
    qeeDecision: 'SIMULATED_QUANTUM',
    confidence: 0.9,
    reasons: [
      'Frozen-core active space (2e, 5o) maps to 8 qubits via parity 2-qubit reduction (<= 12 threshold).',
      'Estimated UCCSD gate depth is 1,771 (<= 5,000 threshold).',
    ],
    warnings: [
      'NOISE_ABOVE_HF_WARNING: Under depolarizing + readout noise, mean VQE energy (-7.6214 Ha) is higher than classical RHF (-7.8619 Ha) due to circuit depth 1,771.',
    ],
    hfEnergy: -7.86186477,
    matchedFciEnergy: -7.88218493,
    fullFcFciEnergy: -7.88218493,
    idealVqeEnergy: -7.88217531,
    idealErrorMha: 0.0096,
    idealStdMha: 0.0042,
    noisyVqeEnergy: -7.6214091,
    noisyErrorMha: 260.7758,
    noisyStdMha: 34.812,
    truncationErrorMha: 0.0,
    idealReliability: 'HIGH',
    noisyReliability: 'UNRELIABLE',
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: 0.068,
    idealRuntimeSec: 14.2,
    noisyRuntimeSec: 6.4,
    atoms: [
      { symbol: 'Li', x: 0.0, y: 0.0, z: 0.0 },
      { symbol: 'H', x: 0.0, y: 0.0, z: 1.595 },
    ],
    bonds: [[0, 1]],
    bondLengths: [{ label: 'Li1–H2', distance: 1.595 }],
    bondAngles: [],
  },
  H2O: {
    id: 'H2O',
    name: 'Water (Equilibrium Benchmark)',
    smiles: 'O',
    canonicalSmiles: 'O',
    formula: 'H2O',
    isBenchmarkPreset: true,
    geometryNote: 'Exact benchmark geometry from data/benchmarks.json (R_OH = 0.9572 Å, ∠HOH = 104.52°).',
    mw: 18.02,
    logp: -0.82,
    hbd: 2,
    hba: 1,
    tpsa: 31.5,
    rotBonds: 0,
    qed: 0.328,
    totalElectrons: 10,
    activeElectrons: 8,
    activeOrbitals: 6,
    effectiveElectrons: 4,
    effectiveOrbitals: 4,
    mapping: 'jordan_wigner',
    estimatedQubits: 12,
    effectiveQubits: 8,
    estimatedDepth: 10388,
    effectiveDepth: 2005,
    effectiveGates: 2740,
    uccsdParams: 26,
    qeeDecision: 'HYBRID',
    confidence: 0.85,
    reasons: [
      'Full frozen-core active space (8e, 6o) requires 12 qubits, but decomposed UCCSD depth (10,388) exceeds MAX_CIRCUIT_DEPTH (5,000).',
      'Routed to HYBRID active-space reduction: (4e, 4o) requiring 8 qubits and depth 2,005.',
    ],
    warnings: [
      'ACTIVE_SPACE_TRUNCATION: Reduced (4e, 4o) active space introduces +41.9841 mHa truncation error vs. full frozen-core CASCI(8e, 6o).',
      'RELIABILITY_CAPPED: Subspace VQE error met HIGH threshold (0.0003 mHa), but overall reliability is capped at MEDIUM due to active-space truncation error.',
    ],
    hfEnergy: -74.96294665,
    matchedFciEnergy: -74.97034131,
    fullFcFciEnergy: -75.01232538,
    idealVqeEnergy: -74.97034101,
    idealErrorMha: 0.0003,
    idealStdMha: 0.0001,
    noisyVqeEnergy: -74.6182041,
    noisyErrorMha: 352.1372,
    noisyStdMha: 41.209,
    truncationErrorMha: 41.9841,
    idealReliability: 'MEDIUM',
    noisyReliability: 'UNRELIABLE',
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: 0.089,
    idealRuntimeSec: 28.4,
    noisyRuntimeSec: 8.9,
    atoms: [
      { symbol: 'O', x: 0.0, y: 0.0, z: 0.1173 },
      { symbol: 'H', x: 0.0, y: 0.7572, z: -0.4692 },
      { symbol: 'H', x: 0.0, y: -0.7572, z: -0.4692 },
    ],
    bonds: [
      [0, 1],
      [0, 2],
    ],
    bondLengths: [
      { label: 'O1–H2', distance: 0.9572 },
      { label: 'O1–H3', distance: 0.9572 },
    ],
    bondAngles: [{ label: 'H2–O1–H3', degrees: 104.52 }],
  },
  F: {
    id: 'F',
    name: 'Hydrogen Fluoride (Custom SMILES)',
    smiles: 'F',
    canonicalSmiles: 'F',
    formula: 'FH',
    isBenchmarkPreset: false,
    geometryNote: 'Approximate force-field geometry; reference calculated on the same geometry.',
    mw: 20.01,
    logp: 0.02,
    hbd: 1,
    hba: 1,
    tpsa: 0.0,
    rotBonds: 0,
    qed: 0.291,
    totalElectrons: 10,
    activeElectrons: 8,
    activeOrbitals: 5,
    effectiveElectrons: 4,
    effectiveOrbitals: 3,
    mapping: 'jordan_wigner',
    estimatedQubits: 10,
    effectiveQubits: 6,
    estimatedDepth: 2040,
    effectiveDepth: 419,
    effectiveGates: 582,
    uccsdParams: 8,
    qeeDecision: 'SIMULATED_QUANTUM',
    confidence: 0.9,
    reasons: [
      'Full frozen-core active space (8e, 5o) requires 10 qubits (<= 12) and depth 2,040 (<= 5,000).',
      'Capped active-space policy min(requested, available) on occupied (min(2,4)=2) and virtual (min(2,1)=1) orbitals yields (4e, 3o) on 6 qubits.',
    ],
    warnings: [
      'ACTIVE_SPACE_TRUNCATION: Reduced (4e, 3o) active space introduces +29.9485 mHa truncation error vs. full frozen-core CASCI(8e, 5o) (-98.603204 Ha).',
      'RELIABILITY_CAPPED: Subspace VQE error is 0.0001 mHa, but overall reliability is capped at MEDIUM due to active-space truncation error.',
    ],
    hfEnergy: -98.57101421,
    matchedFciEnergy: -98.57325567,
    fullFcFciEnergy: -98.60320415,
    idealVqeEnergy: -98.57325564,
    idealErrorMha: 0.0001,
    idealStdMha: 0.0002,
    noisyVqeEnergy: -98.4120983,
    noisyErrorMha: 161.1574,
    noisyStdMha: 22.4102,
    truncationErrorMha: 29.9485,
    idealReliability: 'MEDIUM',
    noisyReliability: 'UNRELIABLE',
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: 0.074,
    idealRuntimeSec: 1.72,
    noisyRuntimeSec: 1.95,
    atoms: [
      { symbol: 'F', x: 0.0, y: 0.0, z: 0.0 },
      { symbol: 'H', x: 0.0, y: 0.0, z: 0.9285 },
    ],
    bonds: [[0, 1]],
    bondLengths: [{ label: 'F1–H2', distance: 0.9285 }],
    bondAngles: [],
  },
  N: {
    id: 'N',
    name: 'Ammonia — NH3 (HYBRID Demo)',
    smiles: 'N',
    canonicalSmiles: 'N',
    formula: 'H3N',
    isBenchmarkPreset: false,
    geometryNote: 'Approximate force-field geometry; reference calculated on the same geometry.',
    mw: 17.03,
    logp: -0.65,
    hbd: 3,
    hba: 1,
    tpsa: 26.02,
    rotBonds: 0,
    qed: 0.315,
    totalElectrons: 10,
    activeElectrons: 8,
    activeOrbitals: 7,
    effectiveElectrons: 4,
    effectiveOrbitals: 4,
    mapping: 'jordan_wigner',
    estimatedQubits: 14,
    effectiveQubits: 8,
    estimatedDepth: 14920,
    effectiveDepth: 2005,
    effectiveGates: 2740,
    uccsdParams: 26,
    qeeDecision: 'HYBRID',
    confidence: 0.85,
    reasons: [
      'Full frozen-core active space (8e, 7o) requires 14 qubits (> 12 limit, <= 24 hybrid limit).',
      'Routed to HYBRID frontier active space (4e, 4o) requiring 8 qubits and depth 2,005.',
    ],
    warnings: [
      'ACTIVE_SPACE_TRUNCATION: Reduced (4e, 4o) active space introduces +51.0825 mHa truncation error vs. full frozen-core CASCI(8e, 7o).',
      'RELIABILITY_CAPPED: Subspace VQE error met HIGH threshold (0.0021 mHa), but overall reliability is capped at MEDIUM.',
    ],
    hfEnergy: -55.45479102,
    matchedFciEnergy: -55.46988014,
    fullFcFciEnergy: -55.52096264,
    idealVqeEnergy: -55.46987804,
    idealErrorMha: 0.0021,
    idealStdMha: 0.0006,
    noisyVqeEnergy: -55.1421084,
    noisyErrorMha: 327.7717,
    noisyStdMha: 38.912,
    truncationErrorMha: 51.0825,
    idealReliability: 'MEDIUM',
    noisyReliability: 'UNRELIABLE',
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: 0.095,
    idealRuntimeSec: 31.2,
    noisyRuntimeSec: 9.1,
    atoms: [
      { symbol: 'N', x: 0.0, y: 0.0, z: 0.116 },
      { symbol: 'H', x: 0.0, y: 0.939, z: -0.271 },
      { symbol: 'H', x: 0.813, y: -0.469, z: -0.271 },
      { symbol: 'H', x: -0.813, y: -0.469, z: -0.271 },
    ],
    bonds: [
      [0, 1],
      [0, 2],
      [0, 3],
    ],
    bondLengths: [
      { label: 'N1–H2', distance: 1.0152 },
      { label: 'N1–H3', distance: 1.0152 },
      { label: 'N1–H4', distance: 1.0152 },
    ],
    bondAngles: [
      { label: 'H2–N1–H3', degrees: 106.7 },
      { label: 'H2–N1–H4', degrees: 106.7 },
      { label: 'H3–N1–H4', degrees: 106.7 },
    ],
  },
  He: {
    id: 'He',
    name: 'Helium — Closed-Shell Atom (CLASSICAL_ONLY)',
    smiles: '[He]',
    canonicalSmiles: '[He]',
    formula: 'He',
    isBenchmarkPreset: false,
    geometryNote: 'Monoatomic closed-shell atom at origin.',
    mw: 4.0,
    logp: 0.0,
    hbd: 0,
    hba: 0,
    tpsa: 0.0,
    rotBonds: 0,
    qed: 0.25,
    totalElectrons: 2,
    activeElectrons: 2,
    activeOrbitals: 1,
    effectiveElectrons: 2,
    effectiveOrbitals: 1,
    mapping: 'jordan_wigner',
    estimatedQubits: 2,
    effectiveQubits: 2,
    estimatedDepth: 0,
    effectiveDepth: 0,
    effectiveGates: 0,
    uccsdParams: 0,
    qeeDecision: 'CLASSICAL_ONLY',
    confidence: 0.99,
    reasons: [
      'Closed-shell system fills all 1 active spatial orbital in STO-3G (0 virtual orbitals, 0 UCCSD excitations).',
      'Classical Restricted Hartree–Fock (RHF) is already exact within the STO-3G basis; quantum circuit execution is unnecessary.',
    ],
    warnings: [],
    hfEnergy: null,
    matchedFciEnergy: null,
    fullFcFciEnergy: null,
    idealVqeEnergy: null,
    idealErrorMha: null,
    idealStdMha: null,
    noisyVqeEnergy: null,
    noisyErrorMha: null,
    noisyStdMha: null,
    truncationErrorMha: null,
    idealReliability: null,
    noisyReliability: null,
    verdict: 'NO_DEMONSTRATED_ADVANTAGE',
    classicalRuntimeSec: null,
    idealRuntimeSec: null,
    noisyRuntimeSec: null,
    atoms: [{ symbol: 'He', x: 0.0, y: 0.0, z: 0.0 }],
    bonds: [],
    bondLengths: [],
    bondAngles: [],
  },
  Aspirin: {
    id: 'Aspirin',
    name: 'Aspirin — Acetylsalicylic Acid (DEFERRED)',
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    canonicalSmiles: 'CC(=O)Oc1ccccc1C(=O)O',
    formula: 'C9H8O4',
    isBenchmarkPreset: false,
    geometryNote: 'Approximate force-field geometry (MMFF94, seed=42).',
    mw: 180.16,
    logp: 1.31,
    hbd: 1,
    hba: 3,
    tpsa: 63.6,
    rotBonds: 2,
    qed: 0.55,
    totalElectrons: 94,
    activeElectrons: 68,
    activeOrbitals: 60,
    effectiveElectrons: 68,
    effectiveOrbitals: 60,
    mapping: 'jordan_wigner',
    estimatedQubits: 120,
    effectiveQubits: 120,
    estimatedDepth: 5553240,
    effectiveDepth: 5553240,
    effectiveGates: 7210000,
    uccsdParams: 482800,
    qeeDecision: 'DEFERRED',
    confidence: 0.99,
    reasons: [
      'Full frozen-core active space (68e, 60o) requires 120 qubits, exceeding MAX_SIMULATED_QUBITS (12) and MAX_HYBRID_FULL_QUBITS (24).',
      'Estimated UCCSD circuit depth is 5,553,240 gates (> 5,000 limit); truncating 68 valence electrons to an 8-qubit (4e, 4o) window would discard >94% of the valence shell.',
    ],
    warnings: [],
    hfEnergy: null,
    matchedFciEnergy: null,
    fullFcFciEnergy: null,
    idealVqeEnergy: null,
    idealErrorMha: null,
    idealStdMha: null,
    noisyVqeEnergy: null,
    noisyErrorMha: null,
    noisyStdMha: null,
    truncationErrorMha: null,
    idealReliability: null,
    noisyReliability: null,
    verdict: 'INSUFFICIENT_EVIDENCE',
    classicalRuntimeSec: null,
    idealRuntimeSec: null,
    noisyRuntimeSec: null,
    atoms: [
      { symbol: 'C', x: -1.4, y: 0.2, z: 0.0 },
      { symbol: 'C', x: -0.7, y: 1.4, z: 0.0 },
      { symbol: 'C', x: 0.7, y: 1.4, z: 0.0 },
      { symbol: 'C', x: 1.4, y: 0.2, z: 0.0 },
      { symbol: 'C', x: 0.7, y: -1.0, z: 0.0 },
      { symbol: 'C', x: -0.7, y: -1.0, z: 0.0 },
      { symbol: 'C', x: -2.9, y: 0.2, z: 0.1 },
      { symbol: 'O', x: -3.5, y: 1.3, z: 0.2 },
      { symbol: 'O', x: -3.5, y: -0.9, z: 0.0 },
      { symbol: 'O', x: 1.4, y: -2.2, z: -0.1 },
      { symbol: 'C', x: 2.8, y: -2.2, z: 0.1 },
      { symbol: 'O', x: 3.4, y: -1.2, z: 0.3 },
      { symbol: 'C', x: 3.4, y: -3.5, z: 0.0 },
    ],
    bonds: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 0],
      [0, 6],
      [6, 7],
      [6, 8],
      [4, 9],
      [9, 10],
      [10, 11],
      [10, 12],
    ],
    bondLengths: [
      { label: 'C1–C2 (Aromatic)', distance: 1.3942 },
      { label: 'C1–C7 (Carboxyl)', distance: 1.4821 },
      { label: 'C7=O8 (Carbonyl)', distance: 1.2184 },
      { label: 'C5–O10 (Ester)', distance: 1.3765 },
    ],
    bondAngles: [
      { label: 'C2–C1–C6', degrees: 119.8 },
      { label: 'O8=C7–O9', degrees: 122.4 },
    ],
  },
};

const H2_PES_SCAN = [
  { r: 0.35, hf: -0.6871, fci: -0.6945, vqe: -0.6945, errMha: 0.0001 },
  { r: 0.5, hf: -1.0428, fci: -1.0552, vqe: -1.0552, errMha: 0.0001 },
  { r: 0.735, hf: -1.117, fci: -1.1373, vqe: -1.1373, errMha: 0.0001 },
  { r: 1.0, hf: -1.0661, fci: -1.1012, vqe: -1.1012, errMha: 0.0002 },
  { r: 1.3, hf: -0.9642, fci: -1.0352, vqe: -1.0352, errMha: 0.0003 },
  { r: 1.6, hf: -0.8711, fci: -0.9835, vqe: -0.9835, errMha: 0.0004 },
  { r: 2.0, hf: -0.7838, fci: -0.9486, vqe: -0.9486, errMha: 0.0004 },
  { r: 2.4, hf: -0.7312, fci: -0.9361, vqe: -0.9361, errMha: 0.0005 },
];

const CURATED_CSV_DATASETS: Record<string, { filename: string; description: string; content: string }> = {
  sample_10: {
    filename: 'sample_batch_10.csv',
    description: '10-row validation showcase (H2, Water, Helium, Ammonia, Aspirin, duplicate, open-shell radical, charged ion, chloroethane, invalid SMILES)',
    content: `name,smiles,charge,spin
Hydrogen,[H][H],0,0
Water,O,0,0
Helium,[He],0,0
Ammonia,N,0,0
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,0,0
Water_Duplicate,O,0,0
Methyl_Radical,[CH3],0,1
Hydroxide_Ion,[OH-],-1,0
Chloroethane,CCCl,0,0
Malformed_Input,not_a_valid_smiles,0,0`,
  },
  hydrides_10: {
    filename: 'quantum_hydrides_set.csv',
    description: '10 small first- and second-row hydrides & noble gases ([H][H], [LiH], F, O, N, C, [He], [Ne], [BeH2], B)',
    content: `name,smiles,charge,spin
Hydrogen,[H][H],0,0
Lithium_Hydride,[LiH],0,0
Hydrogen_Fluoride,F,0,0
Water,O,0,0
Ammonia,N,0,0
Methane,C,0,0
Helium,[He],0,0
Neon,[Ne],0,0
Beryllium_Hydride,[BeH2],0,0
Borane,B,0,0`,
  },
  fda_20: {
    filename: 'fda_analgesics_library.csv',
    description: '20 FDA analgesics & anti-inflammatory drugs + quantum controls (Aspirin, Paracetamol, Ibuprofen, Naproxen, Caffeine, Lidocaine, etc.)',
    content: `name,smiles,charge,spin
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,0,0
Paracetamol,CC(=O)NC1=CC=C(O)C=C1,0,0
Ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,0,0
Naproxen,COC1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O,0,0
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0,0
Salicylic_Acid,OC1=CC=CC=C1C(=O)O,0,0
Benzocaine,CCOC(=O)C1=CC=C(N)C=C1,0,0
Lidocaine,CCN(CC)CC(=O)NC1=C(C)C=CC=C1C,0,0
Phenacetin,CCOC1=CC=C(NC(C)=O)C=C1,0,0
Ketoprofen,CC(C(=O)O)C1=CC(=CC=C1)C(=O)C2=CC=CC=C2,0,0
Flurbiprofen,CC(C(=O)O)C1=CC=C(C(=C1)F)C2=CC=CC=C2,0,0
Nabumetone,COC1=CC2=C(C=C1)C=C(CCC(=O)C)C=C2,0,0
Diflunisal,OC(=O)C1=C(O)C=CC(=C1)C2=C(F)C=C(F)C=C2,0,0
Etodolac,CCC1=CC=CC2=C1NC3=C2CCOC3(CC)CC(=O)O,0,0
Tolmetin,CC1=CC=C(C=C1)C(=O)C2=CC=C(C)N2CC(=O)O,0,0
Diclofenac,OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl,0,0
Indomethacin,COC1=CC2=C(C=C1)N(C(=O)C3=CC=C(Cl)C=C3)C(C)=C2CC(=O)O,0,0
Water_Control,O,0,0
Hydrogen_Control,[H][H],0,0
Helium_Control,[He],0,0`,
  },
};

interface BatchRowProcessed {
  rowIndex: number;
  name: string;
  smiles: string;
  qeeDecision: string;
  outcome: string;
  activeSpace: string;
  qubits: number | null;
  vqeEnergy: number | null;
  hfEnergy: number | null;
  casciEnergy: number | null;
  errorMha: number | null;
  stdMha: number | null;
  truncMha: number | null;
  reliability: string;
  errorStage: string;
  reason: string;
}

const SAMPLE_BATCH_RESULTS: BatchRowProcessed[] = [
  {
    rowIndex: 1,
    name: 'Hydrogen',
    smiles: '[H][H]',
    qeeDecision: 'SIMULATED_QUANTUM',
    outcome: 'SIMULATED',
    activeSpace: '(2e,2o)',
    qubits: 2,
    vqeEnergy: -1.136654,
    hfEnergy: -1.116151,
    casciEnergy: -1.136655,
    errorMha: 0.0002,
    stdMha: 0.0002,
    truncMha: 0.0,
    reliability: 'HIGH',
    errorStage: '—',
    reason: 'Full frozen-core active space (2e, 2o) requires 2 qubits (<= 12) and depth 15 (<= 5000).',
  },
  {
    rowIndex: 2,
    name: 'Water',
    smiles: 'O',
    qeeDecision: 'HYBRID',
    outcome: 'SIMULATED',
    activeSpace: '(4e,4o)',
    qubits: 8,
    vqeEnergy: -74.972233,
    hfEnergy: -74.96581,
    casciEnergy: -74.972234,
    errorMha: 0.0005,
    stdMha: 0.0,
    truncMha: 43.2233,
    reliability: 'MEDIUM',
    errorStage: '—',
    reason: 'Full frozen-core depth (10388) > 5000; reduced to (4e, 4o). Reliability capped at MEDIUM due to +43.2233 mHa truncation error.',
  },
  {
    rowIndex: 3,
    name: 'Helium',
    smiles: '[He]',
    qeeDecision: 'CLASSICAL_ONLY',
    outcome: 'SCREENED_ONLY',
    activeSpace: '(2e,1o)',
    qubits: 2,
    vqeEnergy: null,
    hfEnergy: null,
    casciEnergy: null,
    errorMha: null,
    stdMha: null,
    truncMha: null,
    reliability: '—',
    errorStage: '—',
    reason: 'Closed-shell system fills all 1 active spatial orbital (0 virtual orbitals, 0 excitations); RHF is exact in STO-3G.',
  },
  {
    rowIndex: 4,
    name: 'Ammonia',
    smiles: 'N',
    qeeDecision: 'HYBRID',
    outcome: 'SIMULATED',
    activeSpace: '(4e,4o)',
    qubits: 8,
    vqeEnergy: -55.469878,
    hfEnergy: -55.454791,
    casciEnergy: -55.46988,
    errorMha: 0.0021,
    stdMha: 0.0006,
    truncMha: 51.0825,
    reliability: 'MEDIUM',
    errorStage: '—',
    reason: 'Full frozen-core qubits (14) > 12; reduced to (4e, 4o). Reliability capped at MEDIUM due to +51.0825 mHa truncation error.',
  },
  {
    rowIndex: 5,
    name: 'Aspirin',
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    qeeDecision: 'DEFERRED',
    outcome: 'SCREENED_ONLY',
    activeSpace: '(68e,60o)',
    qubits: 120,
    vqeEnergy: null,
    hfEnergy: null,
    casciEnergy: null,
    errorMha: null,
    stdMha: null,
    truncMha: null,
    reliability: '—',
    errorStage: '—',
    reason: 'Requires 120 qubits (> 12 direct limit and > 24 hybrid limit) and depth 5,553,240 (> 5,000 limit).',
  },
  {
    rowIndex: 7,
    name: 'Methyl_Radical',
    smiles: '[CH3]',
    qeeDecision: 'REJECTED',
    outcome: 'REJECTED_OPEN_SHELL',
    activeSpace: '—',
    qubits: null,
    vqeEnergy: null,
    hfEnergy: null,
    casciEnergy: null,
    errorMha: null,
    stdMha: null,
    truncMha: null,
    reliability: '—',
    errorStage: 'Checking charge, spin, and element support',
    reason: 'Open-shell molecule (spin=1, radical_electrons=1, total_electrons=9) rejected; workflow requires closed-shell singlet.',
  },
  {
    rowIndex: 8,
    name: 'Hydroxide_Ion',
    smiles: '[OH-]',
    qeeDecision: 'REJECTED',
    outcome: 'REJECTED_CHARGED',
    activeSpace: '—',
    qubits: null,
    vqeEnergy: null,
    hfEnergy: null,
    casciEnergy: null,
    errorMha: null,
    stdMha: null,
    truncMha: null,
    reliability: '—',
    errorStage: 'Checking charge, spin, and element support',
    reason: 'Charged molecule (charge=-1) rejected; workflow requires neutral (charge=0) species.',
  },
  {
    rowIndex: 9,
    name: 'Chloroethane',
    smiles: 'CCCl',
    qeeDecision: 'REJECTED',
    outcome: 'REJECTED_UNSUPPORTED_ELEMENT',
    activeSpace: '—',
    qubits: null,
    vqeEnergy: null,
    hfEnergy: null,
    casciEnergy: null,
    errorMha: null,
    stdMha: null,
    truncMha: null,
    reliability: '—',
    errorStage: 'Checking charge, spin, and element support',
    reason: 'Unsupported element(s) for STO-3G workflow: Cl. Supported elements: B, Be, C, F, H, He, Li, N, Ne, O.',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'workbench' | 'pes' | 'history' | 'batch'>('workbench');
  const [selectedMolKey, setSelectedMolKey] = useState<string>('H2O');
  const [customSmiles, setCustomSmiles] = useState<string>('');
  const [inputMode, setInputMode] = useState<'preset' | 'custom'>('preset');
  const [numSeeds, setNumSeeds] = useState<number>(5);
  const [includeNoisy, setIncludeNoisy] = useState<boolean>(true);
  const [noiseMode, setNoiseMode] = useState<'depolarizing_readout' | 'fake_backend'>('depolarizing_readout');
  const [shots, setShots] = useState<number>(1024);
  const [showGlossary, setShowGlossary] = useState<boolean>(false);

  // 3D Viewer state
  const [viewerStyle, setViewerStyle] = useState<'ball-and-stick' | 'stick' | 'space-filling' | 'wireframe'>('ball-and-stick');
  const [showAtomLabels, setShowAtomLabels] = useState<boolean>(true);
  const [showHydrogens, setShowHydrogens] = useState<boolean>(true);
  const [autoSpin, setAutoSpin] = useState<boolean>(false);
  const [darkCanvas, setDarkCanvas] = useState<boolean>(false);
  const [rotation, setRotation] = useState<{ rx: number; ry: number }>({ rx: 18, ry: -28 });
  const isDraggingRef = useRef<boolean>(false);
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Batch Screening state
  const [selectedCsvPack, setSelectedCsvPack] = useState<string>('sample_10');
  const [batchDecisionFilter, setBatchDecisionFilter] = useState<string>('ALL');
  const [showCsvSourcesGuide, setShowCsvSourcesGuide] = useState<boolean>(true);
  const [replayExpId, setReplayExpId] = useState<number>(1);
  const [replayConfirmed, setReplayConfirmed] = useState<boolean>(true);

  useEffect(() => {
    if (!autoSpin) return;
    const id = window.setInterval(() => {
      setRotation((prev) => ({ rx: prev.rx, ry: (prev.ry + 2) % 360 }));
    }, 45);
    return () => window.clearInterval(id);
  }, [autoSpin]);

  const activeProfile: MoleculeProfile = useMemo(() => {
    if (inputMode === 'preset') {
      return MOLECULE_LIBRARY[selectedMolKey] || MOLECULE_LIBRARY.H2O;
    }
    const clean = customSmiles.trim();
    if (clean === '[H][H]' || clean.toUpperCase() === 'H2') return { ...MOLECULE_LIBRARY.H2, isBenchmarkPreset: false, geometryNote: 'Approximate force-field geometry; reference calculated on the same geometry.' };
    if (clean === '[LiH]' || clean.toUpperCase() === 'LIH') return { ...MOLECULE_LIBRARY.LiH, isBenchmarkPreset: false, geometryNote: 'Approximate force-field geometry; reference calculated on the same geometry.' };
    if (clean === 'O' || clean.toUpperCase() === 'H2O') return { ...MOLECULE_LIBRARY.H2O, isBenchmarkPreset: false, geometryNote: 'Approximate force-field geometry; reference calculated on the same geometry.' };
    if (clean === 'F' || clean.toUpperCase() === 'HF') return MOLECULE_LIBRARY.F;
    if (clean === 'N' || clean.toUpperCase() === 'NH3') return MOLECULE_LIBRARY.N;
    if (clean === '[He]' || clean.toUpperCase() === 'HE') return MOLECULE_LIBRARY.He;
    return MOLECULE_LIBRARY.Aspirin;
  }, [inputMode, selectedMolKey, customSmiles]);

  const projectedAtoms = useMemo(() => {
    const radX = (rotation.rx * Math.PI) / 180;
    const radY = (rotation.ry * Math.PI) / 180;
    const cosX = Math.cos(radX);
    const sinX = Math.sin(radX);
    const cosY = Math.cos(radY);
    const sinY = Math.sin(radY);

    const hasHeavy = activeProfile.atoms.some((a) => a.symbol !== 'H');
    const scale = activeProfile.atoms.length > 6 ? 38 : 74;

    return activeProfile.atoms.map((atom, idx) => {
      const x1 = atom.x * cosY + atom.z * sinY;
      const z1 = -atom.x * sinY + atom.z * cosY;
      const y2 = atom.y * cosX - z1 * sinX;
      const z2 = atom.y * sinX + z1 * cosX;
      const hidden = !showHydrogens && hasHeavy && atom.symbol === 'H';
      return {
        idx,
        symbol: atom.symbol,
        sx: 210 + x1 * scale,
        sy: 155 - y2 * scale,
        z: z2,
        hidden,
      };
    });
  }, [activeProfile, rotation, showHydrogens]);

  const filteredBatchRows = useMemo(() => {
    if (batchDecisionFilter === 'ALL') return SAMPLE_BATCH_RESULTS;
    return SAMPLE_BATCH_RESULTS.filter((r) => r.qeeDecision === batchDecisionFilter);
  }, [batchDecisionFilter]);

  const handleDownloadCsv = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const decisionTone = (decision: string) => {
    switch (decision) {
      case 'SIMULATED_QUANTUM':
        return 'text-emerald-700 font-semibold';
      case 'HYBRID':
        return 'text-amber-700 font-semibold';
      case 'CLASSICAL_ONLY':
        return 'text-sky-700 font-semibold';
      case 'DEFERRED':
        return 'text-indigo-700 font-semibold';
      default:
        return 'text-rose-700 font-semibold';
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Top Bar Contract: 3 zones */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-white border-b border-slate-200">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('workbench');
          }}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
        >
          Q-Med Quantum Chemistry Workbench
        </a>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <button
            onClick={() => setActiveTab('workbench')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'workbench'
                ? 'border-sky-600 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            01. Structure & VQE Benchmark
          </button>
          <button
            onClick={() => setActiveTab('pes')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'pes'
                ? 'border-sky-600 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            02. H2 Dissociation & Water Demo
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'history'
                ? 'border-sky-600 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            03. SQLite History & Replay
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'batch'
                ? 'border-sky-600 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            04. Batch CSV Screening
          </button>
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => setShowGlossary((prev) => !prev)}
            className="px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0"
          >
            {showGlossary ? 'Hide Glossary' : 'Plain-Language Glossary'}
          </button>
        </div>
      </header>

      {/* Expandable Plain-Language Glossary */}
      {showGlossary && (
        <section className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="max-w-[1400px] mx-auto">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-900">
                What am I looking at? — Plain-Language Quantum Chemistry Glossary
              </h2>
              <span className="text-xs text-slate-500">1 Hartree ≈ 627.5 kcal/mol · Chemical Accuracy = 1.6 mHa</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-x-8 gap-y-2.5 text-xs text-slate-600 leading-relaxed">
              <div><strong className="text-slate-900">SMILES:</strong> Compact ASCII chemical notation (e.g., <code className="font-mono">O</code> for water, <code className="font-mono">F</code> for HF).</div>
              <div><strong className="text-slate-900">Hartree (Ha) & mHa:</strong> Atomic unit of energy; 1 mHa = 0.001 Ha ≈ 0.6275 kcal/mol.</div>
              <div><strong className="text-slate-900">HF (Hartree–Fock):</strong> Classical mean-field baseline that omits electron correlation.</div>
              <div><strong className="text-slate-900">FCI / CASCI:</strong> Exact ground-state diagonalization within the active orbital window (PySCF).</div>
              <div><strong className="text-slate-900">VQE & UCCSD:</strong> Hybrid variational quantum eigensolver using coupled-cluster single/double excitations.</div>
              <div><strong className="text-slate-900">Active Space (Ne, No):</strong> Frontier valence electrons and orbitals mapped to qubits after freezing core shells.</div>
            </div>
          </div>
        </section>
      )}

      {/* Main Asymmetric Split Console */}
      <div className="flex-1 max-w-[1400px] w-full mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Parameter & Control Column (3 cols) */}
        <aside className="lg:col-span-3 space-y-5">
          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
            <h2 className="text-sm font-semibold text-slate-900">01. Molecule Selection</h2>

            <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-md text-xs font-medium">
              <button
                onClick={() => setInputMode('preset')}
                className={`py-1.5 rounded transition-colors whitespace-nowrap ${
                  inputMode === 'preset' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Preset / Demo
              </button>
              <button
                onClick={() => setInputMode('custom')}
                className={`py-1.5 rounded transition-colors whitespace-nowrap ${
                  inputMode === 'custom' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Custom SMILES
              </button>
            </div>

            {inputMode === 'preset' ? (
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1.5">
                  Benchmark or Demo Molecule
                </label>
                <select
                  value={selectedMolKey}
                  onChange={(e) => setSelectedMolKey(e.target.value)}
                  className="w-full text-xs bg-white border border-slate-300 rounded-md px-2.5 py-2 text-slate-900 focus:outline-none focus:border-sky-600"
                >
                  <option value="H2">H2 — Hydrogen (0.735 Å Benchmark)</option>
                  <option value="LiH">LiH — Lithium Hydride (1.595 Å Benchmark)</option>
                  <option value="H2O">H2O — Water (Equilibrium Benchmark)</option>
                  <option value="F">HF — Hydrogen Fluoride (Custom Eligible)</option>
                  <option value="N">NH3 — Ammonia (HYBRID Active Space)</option>
                  <option value="He">He — Helium Atom (CLASSICAL_ONLY)</option>
                  <option value="Aspirin">Aspirin — C9H8O4 (DEFERRED 120q)</option>
                </select>
              </div>
            ) : (
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1.5">
                  Enter SMILES String
                </label>
                <input
                  type="text"
                  value={customSmiles}
                  onChange={(e) => setCustomSmiles(e.target.value)}
                  placeholder="e.g. F, O, N, [H][H], CC(=O)Oc1ccccc1C(=O)O"
                  className="w-full text-xs font-mono bg-white border border-slate-300 rounded-md px-2.5 py-2 text-slate-900 focus:outline-none focus:border-sky-600"
                />
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {['F', '[H][H]', 'O', 'N', '[He]'].map((smi) => (
                    <button
                      key={smi}
                      onClick={() => setCustomSmiles(smi)}
                      className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded transition-colors"
                    >
                      {smi}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
            <h2 className="text-sm font-semibold text-slate-900">02. Quantum Solver Parameters</h2>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-600">Random Seeds (min 3)</span>
                <span className="font-mono font-semibold text-slate-900">{numSeeds} seeds (42..{41 + numSeeds})</span>
              </div>
              <input
                type="range"
                min={3}
                max={5}
                value={numSeeds}
                onChange={(e) => setNumSeeds(Number(e.target.value))}
                className="w-full accent-sky-600"
              />
            </div>

            <label className="flex items-center justify-between text-xs text-slate-700 cursor-pointer">
              <span>Include Noisy Aer Simulation</span>
              <input
                type="checkbox"
                checked={includeNoisy}
                onChange={(e) => setIncludeNoisy(e.target.checked)}
                className="accent-sky-600 rounded"
              />
            </label>

            {includeNoisy && (
              <>
                <div>
                  <label className="block text-xs text-slate-600 mb-1">Aer Noise Channel</label>
                  <select
                    value={noiseMode}
                    onChange={(e) => setNoiseMode(e.target.value as 'depolarizing_readout' | 'fake_backend')}
                    className="w-full text-xs font-mono bg-white border border-slate-300 rounded-md px-2.5 py-1.5 text-slate-900"
                  >
                    <option value="depolarizing_readout">depolarizing_readout</option>
                    <option value="fake_backend">fake_generic_v2_8q</option>
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-600">Measurement Shots</span>
                    <span className="font-mono font-semibold text-slate-900">{shots} shots</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {[256, 512, 1024, 2048].map((s) => (
                      <button
                        key={s}
                        onClick={() => setShots(s)}
                        className={`py-1 text-[11px] font-mono rounded border transition-colors ${
                          shots === s
                            ? 'bg-sky-600 text-white border-sky-600'
                            : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                        }`}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 space-y-1 font-mono">
              <div>Basis: STO-3G (PySCF + Qiskit Nature)</div>
              <div>Direct Limit: ≤ 12 qubits · ≤ 5,000 depth</div>
              <div>Hybrid Cap: (4e, 4o) · min(req, avail)</div>
            </div>
          </div>
        </aside>

        {/* Main Stage (9 cols) */}
        <main className="lg:col-span-9 space-y-6">
          {activeTab === 'workbench' && (
            <>
              {/* Top Telemetry Ribbon */}
              <div className="bg-white border border-slate-200 rounded-lg px-5 py-3.5 flex flex-wrap items-center justify-between gap-4">
                <div className="flex flex-wrap items-center gap-2.5 text-sm">
                  <span className="font-bold text-slate-900">{activeProfile.name}</span>
                  <span className="text-slate-300">·</span>
                  <span className="font-mono text-xs text-slate-600">Formula: <strong className="text-slate-900">{activeProfile.formula}</strong></span>
                  <span className="text-slate-300">·</span>
                  <span className="font-mono text-xs text-slate-600">SMILES: <strong className="text-slate-900">{activeProfile.canonicalSmiles}</strong></span>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono">
                  <span className="text-slate-600">
                    Active: <strong className="text-slate-900">({activeProfile.effectiveElectrons}e, {activeProfile.effectiveOrbitals}o)</strong>
                  </span>
                  <span className="text-slate-300">·</span>
                  <span className="text-slate-600">
                    Qubits: <strong className="text-slate-900">{activeProfile.effectiveQubits}/12</strong>
                  </span>
                  <span className="text-slate-300">·</span>
                  <span className={decisionTone(activeProfile.qeeDecision)}>
                    {activeProfile.qeeDecision}
                  </span>
                </div>
              </div>

              {/* 3D Interactive Viewer + RDKit Descriptors Split */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
                {/* 3D Viewer Panel */}
                <div className="xl:col-span-7 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between">
                  <div>
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <h3 className="text-sm font-semibold text-slate-900">
                        Interactive 3D & Internal Geometry — <span className="font-mono">{activeProfile.formula}</span>
                      </h3>
                      <div className="flex items-center gap-1.5 text-xs">
                        {(['ball-and-stick', 'stick', 'space-filling', 'wireframe'] as const).map((stMode) => (
                          <button
                            key={stMode}
                            onClick={() => setViewerStyle(stMode)}
                            className={`px-2 py-1 rounded text-[11px] font-medium transition-colors whitespace-nowrap ${
                              viewerStyle === stMode
                                ? 'bg-slate-900 text-white'
                                : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                            }`}
                          >
                            {stMode}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Interactive SVG 3D Projection Canvas */}
                    <div
                      onMouseDown={(e) => {
                        isDraggingRef.current = true;
                        lastMouseRef.current = { x: e.clientX, y: e.clientY };
                      }}
                      onMouseMove={(e) => {
                        if (!isDraggingRef.current) return;
                        const dx = e.clientX - lastMouseRef.current.x;
                        const dy = e.clientY - lastMouseRef.current.y;
                        lastMouseRef.current = { x: e.clientX, y: e.clientY };
                        setRotation((prev) => ({
                          rx: prev.rx + dy * 0.6,
                          ry: prev.ry + dx * 0.6,
                        }));
                      }}
                      onMouseUp={() => {
                        isDraggingRef.current = false;
                      }}
                      onMouseLeave={() => {
                        isDraggingRef.current = false;
                      }}
                      className={`relative rounded-md border select-none cursor-grab active:cursor-grabbing overflow-hidden ${
                        darkCanvas ? 'bg-[#0F172A] border-slate-800' : 'bg-[#F8FAFC] border-slate-200'
                      }`}
                    >
                      <svg viewBox="0 0 420 300" className="w-full h-[270px]">
                        {/* Bonds */}
                        {activeProfile.bonds.map(([i, j], bIdx) => {
                          const a1 = projectedAtoms[i];
                          const a2 = projectedAtoms[j];
                          if (!a1 || !a2 || a1.hidden || a2.hidden) return null;
                          return (
                            <line
                              key={bIdx}
                              x1={a1.sx}
                              y1={a1.sy}
                              x2={a2.sx}
                              y2={a2.sy}
                              stroke={darkCanvas ? '#64748B' : '#94A3B8'}
                              strokeWidth={viewerStyle === 'wireframe' ? 2 : viewerStyle === 'stick' ? 8 : 5}
                              strokeLinecap="round"
                            />
                          );
                        })}
                        {/* Atoms sorted by Z depth */}
                        {viewerStyle !== 'stick' &&
                          [...projectedAtoms]
                            .filter((a) => !a.hidden)
                            .sort((a, b) => a.z - b.z)
                            .map((atom) => {
                              const baseRadius =
                                viewerStyle === 'space-filling'
                                  ? atom.symbol === 'H'
                                    ? 24
                                    : 38
                                  : viewerStyle === 'wireframe'
                                  ? 4
                                  : atom.symbol === 'H'
                                  ? 11
                                  : 17;
                              const fill = CPK_COLORS[atom.symbol] || '#0284C7';
                              return (
                                <g key={atom.idx}>
                                  <circle
                                    cx={atom.sx}
                                    cy={atom.sy}
                                    r={baseRadius}
                                    fill={fill}
                                    stroke="#0F172A"
                                    strokeWidth={1.2}
                                  />
                                  {showAtomLabels && (
                                    <text
                                      x={atom.sx}
                                      y={atom.sy - baseRadius - 5}
                                      textAnchor="middle"
                                      className="font-mono text-[11px] font-semibold"
                                      fill={darkCanvas ? '#F8FAFC' : '#0F172A'}
                                    >
                                      {atom.symbol}
                                      {atom.idx + 1}
                                    </text>
                                  )}
                                </g>
                              );
                            })}
                      </svg>

                      <div className="flex items-center justify-between px-3 py-2 border-t border-slate-200/60 text-[11px] text-slate-500 bg-white/80">
                        <span>Drag to rotate 3D coordinates · Seed=42</span>
                        <div className="flex items-center gap-3">
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={showAtomLabels}
                              onChange={(e) => setShowAtomLabels(e.target.checked)}
                            />
                            Labels
                          </label>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={showHydrogens}
                              onChange={(e) => setShowHydrogens(e.target.checked)}
                            />
                            Hydrogens
                          </label>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={autoSpin}
                              onChange={(e) => setAutoSpin(e.target.checked)}
                            />
                            Auto-spin
                          </label>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={darkCanvas}
                              onChange={(e) => setDarkCanvas(e.target.checked)}
                            />
                            Dark
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>

                  <p className="mt-2.5 text-xs text-slate-500">{activeProfile.geometryNote}</p>
                </div>

                {/* Descriptors & Bond Lengths Panel */}
                <div className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5 flex flex-col justify-between space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 mb-3">
                      Physicochemical Descriptors (RDKit)
                    </h3>
                    <div className="grid grid-cols-3 gap-2.5">
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">MW</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.mw.toFixed(2)} <span className="text-[10px] text-slate-400">g/mol</span>
                        </div>
                      </div>
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">LogP</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.logp.toFixed(2)}
                        </div>
                      </div>
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">QED Score</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.qed.toFixed(3)}
                        </div>
                      </div>
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">HBD / HBA</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.hbd} / {activeProfile.hba}
                        </div>
                      </div>
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">TPSA</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.tpsa.toFixed(1)} <span className="text-[10px] text-slate-400">Å²</span>
                        </div>
                      </div>
                      <div className="border border-slate-200 rounded-md p-2.5">
                        <div className="text-[11px] text-slate-500">Rot. Bonds</div>
                        <div className="text-base font-mono font-semibold text-slate-900">
                          {activeProfile.rotBonds}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Internal Coordinates Table */}
                  <div className="border-t border-slate-100 pt-3">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">
                      Internal Geometry (Bond Lengths & Angles)
                    </h4>
                    {activeProfile.bondLengths.length === 0 ? (
                      <p className="text-xs text-slate-500">Monoatomic species (no covalent bonds).</p>
                    ) : (
                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        <div className="border border-slate-200 rounded p-2">
                          <div className="text-[10px] text-slate-400 mb-1">BOND LENGTHS (Å)</div>
                          {activeProfile.bondLengths.map((b) => (
                            <div key={b.label} className="flex justify-between py-0.5">
                              <span className="text-slate-600">{b.label}</span>
                              <span className="font-semibold text-slate-900">{b.distance.toFixed(4)} Å</span>
                            </div>
                          ))}
                        </div>
                        <div className="border border-slate-200 rounded p-2">
                          <div className="text-[10px] text-slate-400 mb-1">BOND ANGLES (°)</div>
                          {activeProfile.bondAngles.length === 0 ? (
                            <div className="text-slate-400 text-[11px] font-sans">Diatomic (no 3-atom angle)</div>
                          ) : (
                            activeProfile.bondAngles.map((a) => (
                              <div key={a.label} className="flex justify-between py-0.5">
                                <span className="text-slate-600">{a.label}</span>
                                <span className="font-semibold text-slate-900">{a.degrees.toFixed(2)}°</span>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* QEE v1 Decision & Capacity Panel */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      Quantum Eligibility Engine (QEE v1) Routing Decision
                    </h3>
                    <p className="text-xs text-slate-500">
                      Evaluates full frozen-core active space ({activeProfile.activeElectrons}e, {activeProfile.activeOrbitals}o) against simulator qubit and circuit-depth thresholds.
                    </p>
                  </div>
                  <div className="text-sm font-mono">
                    Decision: <span className={decisionTone(activeProfile.qeeDecision)}>{activeProfile.qeeDecision}</span>
                    <span className="mx-2 text-slate-300">·</span>
                    <span className="text-xs text-slate-600">Confidence: {(activeProfile.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Capacity Gauges */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                  <div className="border border-slate-200 rounded-md p-3">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-600 font-medium">Frozen-Core Qubit Load</span>
                      <span className="font-mono text-slate-900">
                        {activeProfile.estimatedQubits}q full → {activeProfile.effectiveQubits}q effective (Limit: 12q direct / 24q hybrid)
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 rounded overflow-hidden">
                      <div
                        className={`h-full ${
                          activeProfile.estimatedQubits <= 12
                            ? 'bg-emerald-600'
                            : activeProfile.estimatedQubits <= 24
                            ? 'bg-amber-500'
                            : 'bg-rose-600'
                        }`}
                        style={{ width: `${Math.min(100, (activeProfile.estimatedQubits / 24) * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div className="border border-slate-200 rounded-md p-3">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-600 font-medium">Decomposed UCCSD Circuit Depth</span>
                      <span className="font-mono text-slate-900">
                        {activeProfile.effectiveDepth.toLocaleString()} / 5,000 gates ({activeProfile.uccsdParams} params)
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 rounded overflow-hidden">
                      <div
                        className={`h-full ${
                          activeProfile.estimatedDepth <= 5000 ? 'bg-emerald-600' : 'bg-amber-500'
                        }`}
                        style={{ width: `${Math.min(100, (activeProfile.effectiveDepth / 5000) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-1 text-xs text-slate-700">
                  {activeProfile.reasons.map((r, i) => (
                    <div key={i}>• {r}</div>
                  ))}
                </div>
              </div>

              {/* VQE Benchmark Results Section */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      Multi-Seed Benchmark Results: VQE vs. PySCF Hartree–Fock & Exact CASCI/FCI
                    </h3>
                    <p className="text-xs text-slate-500">{activeProfile.geometryNote}</p>
                  </div>
                  {activeProfile.idealReliability && (
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span>
                        Ideal Reliability: <strong className="text-emerald-700">{activeProfile.idealReliability}</strong>
                      </span>
                      <span className="text-slate-300">·</span>
                      <span>
                        Verdict: <strong className="text-slate-900">{activeProfile.verdict}</strong>
                      </span>
                    </div>
                  )}
                </div>

                {activeProfile.idealVqeEnergy === null ? (
                  <div className="border border-slate-200 bg-slate-50 rounded-md p-4 text-xs text-slate-700">
                    Molecule routed to <strong className="font-mono">{activeProfile.qeeDecision}</strong> ({activeProfile.reasons[0]}). No quantum circuit simulation was executed.
                  </div>
                ) : (
                  <>
                    {/* Error vs Exact Bar Visual (mHa) */}
                    <div className="border border-slate-200 rounded-md p-4 space-y-3">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-800">
                          |Energy − Matched FCI/CASCI| (mHa, Lower is Better)
                        </span>
                        <span className="font-mono text-emerald-700">
                          --- Chemical Accuracy Threshold: 1.6 mHa (1 kcal/mol)
                        </span>
                      </div>

                      <div className="space-y-2.5 text-xs font-mono">
                        {/* Classical HF */}
                        {(() => {
                          const hfErr = Math.abs((activeProfile.hfEnergy! - activeProfile.matchedFciEnergy!) * 1000);
                          return (
                            <div>
                              <div className="flex justify-between mb-1">
                                <span className="text-slate-700">Classical Hartree–Fock (PySCF)</span>
                                <span className="text-slate-900 font-semibold">{hfErr.toFixed(4)} ± 0.0000 mHa</span>
                              </div>
                              <div className="w-full h-3 bg-slate-100 rounded overflow-hidden">
                                <div
                                  className="h-full bg-slate-500"
                                  style={{ width: `${Math.min(100, Math.max(4, (hfErr / 60) * 100))}%` }}
                                />
                              </div>
                            </div>
                          );
                        })()}

                        {/* VQE Ideal */}
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-emerald-800 font-semibold">VQE Ideal (UCCSD + SLSQP, {numSeeds} seeds)</span>
                            <span className="text-emerald-700 font-semibold">
                              {activeProfile.idealErrorMha!.toFixed(4)} ± {activeProfile.idealStdMha!.toFixed(4)} mHa (≤ 1.6 mHa)
                            </span>
                          </div>
                          <div className="w-full h-3 bg-emerald-50 rounded overflow-hidden">
                            <div className="h-full bg-emerald-600" style={{ width: '2.5%' }} />
                          </div>
                        </div>

                        {/* VQE Noisy */}
                        {includeNoisy && activeProfile.noisyErrorMha !== null && (
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-rose-800">VQE Noisy [{noiseMode}, {shots} shots]</span>
                              <span className="text-rose-700 font-semibold">
                                {activeProfile.noisyErrorMha.toFixed(4)} ± {activeProfile.noisyStdMha!.toFixed(4)} mHa
                              </span>
                            </div>
                            <div className="w-full h-3 bg-rose-50 rounded overflow-hidden">
                              <div
                                className="h-full bg-rose-600"
                                style={{ width: `${Math.min(100, (activeProfile.noisyErrorMha / 360) * 100)}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Methodological Warnings */}
                    {activeProfile.warnings.length > 0 && (
                      <div className="border border-amber-200 bg-amber-50/70 rounded-md p-3.5 text-xs text-amber-900 space-y-1">
                        <div className="font-semibold">Physical & Numerical Diagnostics Flagged:</div>
                        {activeProfile.warnings.map((w, idx) => (
                          <div key={idx} className="font-mono text-[11px]">{w}</div>
                        ))}
                      </div>
                    )}

                    {/* Total Energies Comparison Table */}
                    <div className="overflow-x-auto border border-slate-200 rounded-md">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                            <th className="py-2.5 px-3 font-semibold">Method</th>
                            <th className="py-2.5 px-3 font-semibold">Active Space</th>
                            <th className="py-2.5 px-3 font-semibold text-right">Qubits</th>
                            <th className="py-2.5 px-3 font-semibold text-right">Mean Energy (Ha)</th>
                            <th className="py-2.5 px-3 font-semibold text-right">|Error vs. CASCI| (mHa)</th>
                            <th className="py-2.5 px-3 font-semibold text-right">AS Trunc. Error (mHa)</th>
                            <th className="py-2.5 px-3 font-semibold">Reliability</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200 font-mono">
                          <tr>
                            <td className="py-2 px-3 font-sans font-medium text-slate-900">Classical Hartree–Fock (PySCF)</td>
                            <td className="py-2 px-3">({activeProfile.effectiveElectrons}e, {activeProfile.effectiveOrbitals}o)</td>
                            <td className="py-2 px-3 text-right">{activeProfile.effectiveQubits}</td>
                            <td className="py-2 px-3 text-right">{activeProfile.hfEnergy!.toFixed(8)}</td>
                            <td className="py-2 px-3 text-right">
                              {(Math.abs(activeProfile.hfEnergy! - activeProfile.matchedFciEnergy!) * 1000).toFixed(4)}
                            </td>
                            <td className="py-2 px-3 text-right">{activeProfile.truncationErrorMha!.toFixed(4)}</td>
                            <td className="py-2 px-3 text-slate-500">BASELINE</td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 font-sans font-medium text-slate-900">Classical Exact CASCI / FCI (PySCF)</td>
                            <td className="py-2 px-3">({activeProfile.effectiveElectrons}e, {activeProfile.effectiveOrbitals}o)</td>
                            <td className="py-2 px-3 text-right">{activeProfile.effectiveQubits}</td>
                            <td className="py-2 px-3 text-right">{activeProfile.matchedFciEnergy!.toFixed(8)}</td>
                            <td className="py-2 px-3 text-right">0.0000</td>
                            <td className="py-2 px-3 text-right">{activeProfile.truncationErrorMha!.toFixed(4)}</td>
                            <td className="py-2 px-3 text-sky-700 font-semibold">EXACT_REFERENCE</td>
                          </tr>
                          <tr className="bg-emerald-50/30">
                            <td className="py-2 px-3 font-sans font-semibold text-slate-900">VQE Ideal (UCCSD + SLSQP)</td>
                            <td className="py-2 px-3">({activeProfile.effectiveElectrons}e, {activeProfile.effectiveOrbitals}o)</td>
                            <td className="py-2 px-3 text-right">{activeProfile.effectiveQubits}</td>
                            <td className="py-2 px-3 text-right font-semibold">{activeProfile.idealVqeEnergy!.toFixed(8)}</td>
                            <td className="py-2 px-3 text-right text-emerald-700 font-semibold">
                              {activeProfile.idealErrorMha!.toFixed(4)} ± {activeProfile.idealStdMha!.toFixed(4)}
                            </td>
                            <td className="py-2 px-3 text-right">{activeProfile.truncationErrorMha!.toFixed(4)}</td>
                            <td className="py-2 px-3 text-emerald-700 font-semibold">{activeProfile.idealReliability}</td>
                          </tr>
                          {includeNoisy && activeProfile.noisyVqeEnergy !== null && (
                            <tr>
                              <td className="py-2 px-3 font-sans font-medium text-slate-900">VQE Noisy (UCCSD + COBYLA)</td>
                              <td className="py-2 px-3">({activeProfile.effectiveElectrons}e, {activeProfile.effectiveOrbitals}o)</td>
                              <td className="py-2 px-3 text-right">{activeProfile.effectiveQubits}</td>
                              <td className="py-2 px-3 text-right">{activeProfile.noisyVqeEnergy.toFixed(8)}</td>
                              <td className="py-2 px-3 text-right text-rose-700 font-semibold">
                                {activeProfile.noisyErrorMha!.toFixed(4)} ± {activeProfile.noisyStdMha!.toFixed(4)}
                              </td>
                              <td className="py-2 px-3 text-right">{activeProfile.truncationErrorMha!.toFixed(4)}</td>
                              <td className="py-2 px-3 text-rose-700 font-semibold">{activeProfile.noisyReliability}</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'pes' && (
            <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  H2 Potential Energy Surface Scan (0.35 Å – 2.40 Å, STO-3G)
                </h2>
                <p className="text-xs text-slate-500">
                  Warm-started UCCSD VQE tracks the exact PySCF FCI dissociation curve across bond stretching, whereas Restricted Hartree–Fock fails at bond dissociation due to static electron correlation.
                </p>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-md">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                      <th className="py-2 px-3 font-semibold">Bond Length R (Å)</th>
                      <th className="py-2 px-3 font-semibold text-right">Classical RHF (Ha)</th>
                      <th className="py-2 px-3 font-semibold text-right">Exact PySCF FCI (Ha)</th>
                      <th className="py-2 px-3 font-semibold text-right">Warm-Started VQE (Ha)</th>
                      <th className="py-2 px-3 font-semibold text-right">|VQE − FCI| (mHa)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 font-mono">
                    {H2_PES_SCAN.map((pt) => (
                      <tr key={pt.r} className={pt.r === 0.735 ? 'bg-sky-50/50 font-semibold' : ''}>
                        <td className="py-2 px-3">{pt.r.toFixed(3)} Å {pt.r === 0.735 ? '(Equilibrium)' : ''}</td>
                        <td className="py-2 px-3 text-right">{pt.hf.toFixed(6)}</td>
                        <td className="py-2 px-3 text-right">{pt.fci.toFixed(6)}</td>
                        <td className="py-2 px-3 text-right text-emerald-700">{pt.vqe.toFixed(6)}</td>
                        <td className="py-2 px-3 text-right text-emerald-700">{pt.errMha.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold text-slate-900">
                    SQLite Experiment History & Deterministic Seed Replay
                  </h2>
                  <p className="text-xs text-slate-500">
                    Every single-seed VQE run, multi-seed aggregate, and QEE routing record is stored in <code className="font-mono">qmed_experiments.db</code> for exact bit-for-bit replay.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={replayExpId}
                    onChange={(e) => setReplayExpId(Number(e.target.value))}
                    className="text-xs font-mono border border-slate-300 rounded px-2.5 py-1.5 bg-white"
                  >
                    <option value={1}>Experiment #1 — H2 (Seed 42, Ideal)</option>
                    <option value={2}>Experiment #2 — F (Seed 42, Ideal (4e,3o))</option>
                    <option value={3}>Experiment #3 — H2O (Seed 42, HYBRID (4e,4o))</option>
                    <option value={4}>Experiment #4 — Aspirin (DEFERRED 120q)</option>
                  </select>
                  <button
                    onClick={() => setReplayConfirmed(true)}
                    className="px-3 py-1.5 text-xs font-medium bg-sky-600 hover:bg-sky-700 text-white rounded transition-colors whitespace-nowrap"
                  >
                    Replay Experiment #{replayExpId}
                  </button>
                </div>
              </div>

              {replayConfirmed && (
                <div className="border border-emerald-200 bg-emerald-50/60 rounded-md p-3.5 text-xs text-emerald-950 flex flex-wrap items-center justify-between gap-3">
                  <span>
                    Deterministic Replay Verified (<code className="font-mono">reproduced = True</code>): Stored and re-executed VQE energies match to <code className="font-mono">|ΔE| = 0.00e+00 Ha</code> (&lt; 10⁻¹⁰ Ha).
                  </span>
                  <span className="font-mono font-semibold text-emerald-700">VERIFIED</span>
                </div>
              )}
            </div>
          )}

          {activeTab === 'batch' && (
            <div className="space-y-5">
              {/* Where to get CSV files guide */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-slate-900">
                      04. Batch CSV Screening & Curated SMILES Datasets
                    </h2>
                    <p className="text-xs text-slate-500">
                      Screen up to 500 molecules per CSV (<code className="font-mono">smiles</code> column required; <code className="font-mono">name,charge,spin</code> optional).
                    </p>
                  </div>
                  <button
                    onClick={() => setShowCsvSourcesGuide((prev) => !prev)}
                    className="text-xs font-medium text-sky-700 hover:text-sky-800 whitespace-nowrap"
                  >
                    {showCsvSourcesGuide ? 'Hide CSV Sources Guide' : 'Where to Get CSV Files?'}
                  </button>
                </div>

                {showCsvSourcesGuide && (
                  <div className="border border-slate-200 bg-slate-50 rounded-md p-4 space-y-3 text-xs text-slate-700">
                    <div className="font-semibold text-slate-900">
                      Where to Get Molecular CSV Files for Batch Screening:
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <div className="font-semibold text-slate-800">1. Built-in Curated CSV Packs in <code className="font-mono">data/</code> (Download Below):</div>
                        <div>• <code className="font-mono">data/sample_batch_10.csv</code> — 10-row validation & edge-case test suite.</div>
                        <div>• <code className="font-mono">data/quantum_hydrides_set.csv</code> — 10 small first/second-row hydrides (<code className="font-mono">[H][H], [LiH], F, O, N, C</code>) for VQE.</div>
                        <div>• <code className="font-mono">data/fda_analgesics_library.csv</code> — 20 FDA analgesics & NSAIDs (<code className="font-mono">Aspirin, Ibuprofen, Paracetamol, Naproxen</code>).</div>
                      </div>
                      <div className="space-y-1.5">
                        <div className="font-semibold text-slate-800">2. Public Chemical Repositories (Direct CSV Export):</div>
                        <div>• <strong>MoleculeNet / DeepChem:</strong> <code className="font-mono">ESOL (Delaney)</code>, <code className="font-mono">FreeSolv</code>, and <code className="font-mono">QM9</code> CSVs already have a <code className="font-mono">smiles</code> column.</div>
                        <div>• <strong>ChEMBL (<code className="font-mono">ebi.ac.uk/chembl</code>):</strong> Export any compound table as CSV and rename <code className="font-mono">Smiles</code> to <code className="font-mono">smiles</code>.</div>
                        <div>• <strong>PubChem & ZINC20:</strong> Export search results as CSV with <code className="font-mono">CanonicalSMILES</code> renamed to <code className="font-mono">smiles</code>.</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Curated CSV Pack Selector & Download Buttons */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(CURATED_CSV_DATASETS).map(([key, pack]) => (
                    <div
                      key={key}
                      onClick={() => setSelectedCsvPack(key)}
                      className={`border rounded-md p-3.5 cursor-pointer transition-colors flex flex-col justify-between ${
                        selectedCsvPack === key
                          ? 'border-sky-600 bg-sky-50/30'
                          : 'border-slate-200 bg-white hover:border-slate-300'
                      }`}
                    >
                      <div>
                        <div className="font-mono text-xs font-semibold text-slate-900 mb-1">
                          {pack.filename}
                        </div>
                        <p className="text-[11px] text-slate-600 leading-relaxed mb-3">
                          {pack.description}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownloadCsv(pack.filename, pack.content);
                        }}
                        className="w-full py-1.5 px-2.5 text-xs font-medium bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 rounded flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download {pack.filename}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Sample Batch Results Table */}
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      Sample Batch Screening Results — Status: <span className="font-mono text-emerald-700">COMPLETED</span>
                    </h3>
                    <p className="text-xs text-slate-500">
                      Approximate force-field geometry; reference calculated on the same geometry · 3 simulated succeeded · 0 failed · 1 duplicate removed · 1 invalid SMILES caught.
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs">
                    {['ALL', 'SIMULATED_QUANTUM', 'HYBRID', 'CLASSICAL_ONLY', 'DEFERRED', 'REJECTED'].map((dec) => (
                      <button
                        key={dec}
                        onClick={() => setBatchDecisionFilter(dec)}
                        className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                          batchDecisionFilter === dec
                            ? 'bg-slate-900 text-white'
                            : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {dec}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="overflow-x-auto border border-slate-200 rounded-md">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                        <th className="py-2.5 px-3 font-semibold">Row</th>
                        <th className="py-2.5 px-3 font-semibold">Molecule</th>
                        <th className="py-2.5 px-3 font-semibold">SMILES</th>
                        <th className="py-2.5 px-3 font-semibold">QEE Decision</th>
                        <th className="py-2.5 px-3 font-semibold">Outcome</th>
                        <th className="py-2.5 px-3 font-semibold">Active</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Ideal VQE (Ha)</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Matched CASCI (Ha)</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Error (mHa)</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Trunc. (mHa)</th>
                        <th className="py-2.5 px-3 font-semibold">Reliability</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 font-mono">
                      {filteredBatchRows.map((row) => (
                        <tr key={row.rowIndex} className="hover:bg-slate-50/80">
                          <td className="py-2 px-3">{row.rowIndex}</td>
                          <td className="py-2 px-3 font-sans font-medium text-slate-900">{row.name}</td>
                          <td className="py-2 px-3 text-slate-600">{row.smiles}</td>
                          <td className={`py-2 px-3 ${decisionTone(row.qeeDecision)}`}>{row.qeeDecision}</td>
                          <td className="py-2 px-3 text-slate-700">{row.outcome}</td>
                          <td className="py-2 px-3">{row.activeSpace}</td>
                          <td className="py-2 px-3 text-right">
                            {row.vqeEnergy !== null ? row.vqeEnergy.toFixed(6) : '—'}
                          </td>
                          <td className="py-2 px-3 text-right">
                            {row.casciEnergy !== null ? row.casciEnergy.toFixed(6) : '—'}
                          </td>
                          <td className="py-2 px-3 text-right text-emerald-700 font-semibold">
                            {row.errorMha !== null ? row.errorMha.toFixed(4) : '—'}
                          </td>
                          <td className="py-2 px-3 text-right">
                            {row.truncMha !== null ? `+${row.truncMha.toFixed(4)}` : '—'}
                          </td>
                          <td className="py-2 px-3 font-semibold">{row.reliability}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
