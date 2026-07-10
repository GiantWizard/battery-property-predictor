# feature engineering, reused from chemical-property-predictor (cells 5-6)
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

IN_PATH = "data/qm9ipea_smiles_ip.csv"
OUT_PATH = "data/descriptors.csv"


def safe_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Descriptors.CalcMolDescriptors(mol)


def main():
    t0 = time.time()
    df = pd.read_csv(IN_PATH)
    n_start = len(df)
    print(f"Input molecules: {n_start}")

    # RDKit 2D descriptors
    desc_series = df["smiles"].apply(safe_descriptors)
    valid_mask = desc_series.notna()
    descriptor_list = pd.DataFrame(desc_series[valid_mask].tolist(), index=df.index[valid_mask])
    df = df.loc[valid_mask]
    df = pd.concat([df, descriptor_list], axis=1)
    print(f"After 2D descriptors: {df.shape[0]} molecules, {df.shape[1]} columns")

    # Morgan fingerprints (radius 2, 2048 bits) -- same as reference repo
    fps_list, fps_indices = [], []
    for index, smi in df["smiles"].items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_list.append(arr)
        fps_indices.append(index)

    fps = pd.DataFrame(fps_list, index=fps_indices, columns=[f"fp_{i}" for i in range(2048)])
    df = df.loc[fps_indices]
    df = pd.concat([df, fps], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"After fingerprints: {df.shape[0]} molecules, {df.shape[1]} columns")

    elapsed = time.time() - t0
    df.to_csv(OUT_PATH, index=False)

    print(f"Retained {df.shape[0]} / {n_start} molecules "
          f"({100 * df.shape[0] / n_start:.1f}%) after featurization")
    print(f"Featurization time: {elapsed:.1f}s")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
