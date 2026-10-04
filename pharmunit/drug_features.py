from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger

if not os.environ.get("PHARMUNIT_RDKIT_VERBOSE"):
    RDLogger.DisableLog("rdApp.warning")
from rdkit import RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

PHARM_FAMILIES: tuple[str, ...] = (
    "Donor", "Acceptor", "Aromatic", "Hydrophobe",
    "PosIonizable", "NegIonizable", "None",
)
PHARM_INDEX: dict[str, int] = {f: i for i, f in enumerate(PHARM_FAMILIES)}
_FAMILY_ALIAS: dict[str, str] = {
    "LumpedHydrophobe": "Hydrophobe",
    "ZnBinder": "None",
}

ATOM_VOCAB: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Other",
)
_ATOM_INDEX: dict[str, int] = {s: i for i, s in enumerate(ATOM_VOCAB)}

FUNCTIONAL_SMARTS: tuple[tuple[str, str], ...] = (
    ("sulfonamide",    "[SX4](=O)(=O)[NX3]"),
    ("sulfonate",      "[SX4](=O)(=O)[OX2H1,OX1-]"),
    ("phosphate",      "[PX4](=O)([OX2H1,OX1-])[OX2H1,OX1-]"),
    ("amide",          "[NX3][CX3](=[OX1])"),
    ("carboxyl",       "[CX3](=O)[OX2H1,OX1-]"),
    ("ester",          "[CX3](=O)[OX2][CX4]"),
    ("carbamate",      "[NX3][CX3](=O)[OX2]"),
    ("urea",           "[NX3][CX3](=O)[NX3]"),
    ("guanidine",      "[NX3][CX3](=[NX2,NX3+])[NX3]"),
    ("nitro",          "[NX3](=O)=O"),
    ("nitrile",        "[CX2]#[NX1]"),
    ("aldehyde",       "[CX3H1](=O)"),
    ("ketone",         "[#6][CX3](=O)[#6]"),
    ("ether",          "[OX2]([CX4])[CX4]"),
    ("hydroxyl",       "[OX2H1][CX4]"),
    ("phenol_oh",      "[OX2H1][c]"),
    ("primary_amine",  "[NX3;H2][CX4]"),
    ("secondary_amine","[NX3;H1]([CX4])[CX4]"),
    ("tertiary_amine", "[NX3]([CX4])([CX4])[CX4]"),
    ("thiol",          "[SX2H1][CX4]"),
    ("halide",         "[F,Cl,Br,I]"),
)


_FEATURE_FACTORY: Optional[ChemicalFeatures.MolChemicalFeatureFactory] = None


def _feature_factory() -> ChemicalFeatures.MolChemicalFeatureFactory:
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _FEATURE_FACTORY


def _atom_pharm_types(mol: Chem.Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    out = np.zeros((n_atoms, len(PHARM_FAMILIES)), dtype=np.float32)
    assigned = np.zeros(n_atoms, dtype=np.int16) - 1

    feats = _feature_factory().GetFeaturesForMol(mol)
    priority = {"Donor": 0, "Acceptor": 1, "Aromatic": 2, "Hydrophobe": 3,
                "PosIonizable": 4, "NegIonizable": 5, "None": 6}

    for feat in feats:
        fam = _FAMILY_ALIAS.get(feat.GetFamily(), feat.GetFamily())
        if fam not in PHARM_INDEX:
            continue
        p_new = priority[fam]
        idx_new = PHARM_INDEX[fam]
        for a in feat.GetAtomIds():
            cur = assigned[a]
            if cur < 0 or p_new < priority[PHARM_FAMILIES[cur]]:
                if cur >= 0:
                    out[a, cur] = 0.0
                out[a, idx_new] = 1.0
                assigned[a] = idx_new

    for a in range(n_atoms):
        if assigned[a] < 0:
            out[a, PHARM_INDEX["None"]] = 1.0
    return out


def _atom_basic_features(mol: Chem.Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    d = len(ATOM_VOCAB) + 5
    out = np.zeros((n_atoms, d), dtype=np.float32)
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        out[i, _ATOM_INDEX.get(sym, _ATOM_INDEX["Other"])] = 1.0
        out[i, len(ATOM_VOCAB) + 0] = float(a.GetDegree())
        out[i, len(ATOM_VOCAB) + 1] = float(a.GetFormalCharge())
        out[i, len(ATOM_VOCAB) + 2] = 1.0 if a.GetIsAromatic() else 0.0
        out[i, len(ATOM_VOCAB) + 3] = 1.0 if a.IsInRing() else 0.0
        out[i, len(ATOM_VOCAB) + 4] = float(a.GetTotalNumHs())
    return out


def atom_feature_dim() -> int:
    return len(ATOM_VOCAB) + 5 + len(PHARM_FAMILIES)


def _fused_ring_systems(mol: Chem.Mol) -> list[set[int]]:
    rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
    systems: list[set[int]] = []
    for r in rings:
        merged = r
        keep: list[set[int]] = []
        for s in systems:
            if merged & s:
                merged = merged | s
            else:
                keep.append(s)
        keep.append(merged)
        systems = keep
    return systems


def _functional_group_units(mol: Chem.Mol, forbidden: set[int]) -> list[set[int]]:
    used: set[int] = set()
    units: list[set[int]] = []
    for _name, smarts in FUNCTIONAL_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            m = set(match)
            if m & forbidden or m & used:
                continue
            units.append(m)
            used |= m
    return units


def _linker_units(mol: Chem.Mol, assigned: set[int]) -> list[set[int]]:
    remaining = [i for i in range(mol.GetNumAtoms()) if i not in assigned]
    if not remaining:
        return []
    remaining_set = set(remaining)
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in remaining:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.add(v)
            for nb in mol.GetAtomWithIdx(v).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining_set and j not in seen:
                    stack.append(j)
        comps.append(comp)
    return comps


def partition_units(mol: Chem.Mol) -> list[set[int]]:
    units: list[set[int]] = []
    assigned: set[int] = set()

    for r in _fused_ring_systems(mol):
        units.append(r)
        assigned |= r

    for fg in _functional_group_units(mol, forbidden=assigned):
        units.append(fg)
        assigned |= fg

    for link in _linker_units(mol, assigned=assigned):
        units.append(link)
        assigned |= link

    total = sum(len(u) for u in units)
    if total != mol.GetNumAtoms():
        raise RuntimeError(
            f"unit partition failed: covered {total} of {mol.GetNumAtoms()} atoms"
        )
    return units


@dataclass
class DrugFeatures:
    smiles: str
    atom_x: np.ndarray
    adj: np.ndarray
    unit_idx: np.ndarray
    unit_pharm: np.ndarray
    n_atoms: int
    n_units: int


def featurize_drug(smiles: str) -> Optional[DrugFeatures]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

    n = mol.GetNumAtoms()

    pharm_atom = _atom_pharm_types(mol)
    basic = _atom_basic_features(mol)
    atom_x = np.concatenate([basic, pharm_atom], axis=1).astype(np.float32)

    adj = np.eye(n, dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0

    units = partition_units(mol)
    unit_idx = np.zeros(n, dtype=np.int32)
    for u_id, atoms in enumerate(units):
        for a in atoms:
            unit_idx[a] = u_id

    U = len(units)
    unit_pharm = np.zeros((U, len(PHARM_FAMILIES)), dtype=np.float32)
    for u_id, atoms in enumerate(units):
        counts = pharm_atom[list(atoms)].sum(axis=0)
        s = counts.sum()
        unit_pharm[u_id] = counts / s if s > 0 else counts

    return DrugFeatures(
        smiles=smiles,
        atom_x=atom_x,
        adj=adj,
        unit_idx=unit_idx,
        unit_pharm=unit_pharm,
        n_atoms=n,
        n_units=U,
    )
