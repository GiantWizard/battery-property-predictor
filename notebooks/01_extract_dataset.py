"""
Step 1: extract SMILES + ionization potential (IP) from QM9IPEA.json
(Weinreich et al. arXiv:2411.00994, Zenodo 10.5281/zenodo.13952172).

The raw JSON has geometry and per-charge-state energies but no SMILES, so we
recover SMILES from the neutral-molecule geometry with RDKit's bond perceiver.
IP = E(cation) - E(neutral) at the neutral geometry, using
PNO-LCCSD(T)-F12B for the neutral and PNO-UCCSD(T)-F12B for the cation.
See RESULTS.md for details on the dataset and this step's spot-check.
"""
import json
import time

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

HARTREE_TO_EV = 27.211386245988

DATA_PATH = "data/QM9IPEA.json"
OUT_PATH = "data/qm9ipea_smiles_ip.csv"

NEUTRAL_LEVEL = "PNO-LCCSD(T)-F12B"
CHARGED_LEVEL = "PNO-UCCSD(T)-F12B"


def xyz_block(symbols, coords):
    lines = [str(len(symbols)), ""]
    for s, c in zip(symbols, coords):
        lines.append(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}")
    return "\n".join(lines)


def smiles_from_geometry(symbols, coords):
    block = xyz_block(symbols, coords)
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main():
    t0 = time.time()
    with open(DATA_PATH) as f:
        d = json.load(f)

    n_total = len(d["QM9_ID"])
    print(f"Raw QM9-IPEA entries: {n_total}")

    e_neutral = d["ENERGY"]["0"][NEUTRAL_LEVEL]
    e_cation = d["ENERGY"]["1"][CHARGED_LEVEL]

    rows = []
    n_geom_fail = 0
    n_energy_missing = 0

    for i in range(n_total):
        en = e_neutral[i]
        ec = e_cation[i]
        if en is None or ec is None or en != en or ec != ec:  # en != en catches NaN
            n_energy_missing += 1
            continue

        smi = smiles_from_geometry(d["SYMBOLS"][i], d["COORDS"][i])
        if smi is None:
            n_geom_fail += 1
            continue

        ip_ev = (ec - en) * HARTREE_TO_EV
        rows.append(
            {
                "qm9_id": d["QM9_ID"][i],
                "smiles": smi,
                "ionization_potential_eV": ip_ev,
            }
        )

    elapsed = time.time() - t0

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"Missing neutral/cation energy at {NEUTRAL_LEVEL}/{CHARGED_LEVEL}: {n_energy_missing}")
    print(f"Geometry -> SMILES perception failures: {n_geom_fail}")
    print(f"Retained rows: {len(df)} / {n_total}")
    print(f"Extraction time: {elapsed:.1f}s")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
