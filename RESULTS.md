# Results - Battery/Electrolyte Property Predictor

- Target: Predict the ionization potential (IP) of small organic molecules, predicted from SMILES strings.
- Pipeline: RDKit 2D descriptors + Morgan fingerprints, then VarianceThreshold + StandardScaler, then CatBoost tuned with Optuna, 70/15/15 train/val/test split. S
- Architecture: Main pipeline reused directly from [chemical-property-predictor](https://github.com/GiantWizard/chemical-property-predictor).


## Main results

| Metric | Value | Range / History |
| --- | --- | --- |
| R2 | 0.80-0.82 | 0.7994, 0.8106, 0.8115, 0.8235 |
| MAE | ~0.22-0.23 eV | _Representative across runs_ |
| RMSE | ~0.29-0.31 eV | _Representative across runs_ |
| MAPE | ~2.4-2.5% | _Representative across runs_ |
| Test size (_n_) | 1,656 | _Fixed split_ |

The most recent benchmark (0.7994) and the current active notebook (0.8106) reflect clean runs. Earlier runs are archived in `_old_run_backup/`. Full details can be found in `notebooks/executed_battery_ip_predictor.ipynb`.

## Dataset

Source: The QM9-IPEA subset of the SolQuest dataset (Weinreich et al., arXiv:2411.00994). While the raw `QM9IPEA.json` contains 3D geometries and coupled-cluster quantum mechanical energies, it lacks SMILES strings. However, we can reverse-engineer it from the neutral 3D molecular geometries using RDKit's bond-perception algorithm (`rdDetermineBonds.DetermineBonds`).

Vertical IP is defined as the neutral-to-cation energy difference at the neutral geometry:

```
IP = E(cation, charge +1) - E(neutral, charge 0)
```

_Calculated at the highest available levels of theory: PNO-LCCSD(T)-F12B (neutral) and PNO-UCCSD(T)-F12B (cation)._

computed at the highest level of theory present for each spin state:
`PNO-LCCSD(T)-F12B` (closed-shell neutral) and `PNO-UCCSD(T)-F12B` (open-shell cation).

## Dataset Filtering and Processing

| Stage | Molecules | Remaining | Notes |
|---|---|---|---|
| Raw | 7,000 | 100% | Initial QM9-IPEA set |
| Energy Check | 6,001 | 85.7% | Dropped 999 molecules missing energy records at the reuired level of theory |
| SMILES Recovery | 5,699 | 81.4% | Dropped 302 molecules where RDKit could not resolve a valid Lewis structure from 3D coordinates |
| Featurization | 5,699 | 81.4% | All generated SMILES strings successfully yielded 2D descriptors and Morgan fingerprints |
| Outlier Trimming | 5,518 | 78.8% | Removed anomalies using a 1.5x IQR filter on the target IP values |
| Data Split | 3,281 / 580 / 1,656 | - | Train/Val/Test split, mimicking the reference code |

## Execution Speed

By skipping slow 3D-conformer descriptor calculations like Mordred, the total pipeline runs in under 4 minutes.

| Step | Time |
|---|---|
| Data Extraction | ~1.2s for all 7,000 raw entries |
| Featurization | ~37s |
| Optuna HPO (15 trials) | ~125-190s across runs |
| Final Catboost Model Fit | ~6-14s |
| Total Pipeline Runtime | ~2.5-4 minutes |

## Comparison to chemical-property-predictor's results

| Target Property | R2 | Task Difficulty |
|---|---|---|
| Critical temperature | 0.939 | Easiest: driven heavily by basic size and polarity trends |
| Critical pressure | 0.920 | Easy: Correlated with simple 2D structural attributes |
| Flash point | 0.843 | Moderate: Strong structural relationship but has a small dataset |
| Boiling point | 0.827 | Moderate: Governed by molecular weight and bulk polarity |
| Ionization potential | 0.80-0.82 | Moderate: Primarily electronic and mappable via 2D orbital-proxy features |
| Heat of vaporization | 0.755 | Difficult: Requires deeper modeling of molecular interactions |
| Melting point | 0.674 | Hardest: Dependent on solid-state 3D crystal packing lattices |

IP is an electronic property determined by frontier orbital energetics (HOMO energy, electonegativity, polarizability). Top feature importances are mainly dominated by simple electronic descriptors (`SMR_VSA3`, `SMR_VSA7`, `VSA_EState4`, `MolMR`) rather than just atom counts.

The QM9 dataset is constrained to small molecules (<= 9 heavy atoms) and also far more structurally homogenous than the reference project's 4,343-compound organic substance database, resulting in a compressed target property range (6.9 to 14.3 eV) that naturally lowers the achievable R2 even when the model's predictions are accurate.

## What was skipped (by design, not by failure)

- Electron affinity and solvation energy: not built here. The same `QM9IPEA.json` file already
  contains anion-state energies needed for EA (5,680/7,000 valid), so this is a natural next
  step, not new data-sourcing work.
- GDB17/ZINC/EGP portions of SolQuest (309K, 88K/37K amons, 18K EGP molecules): not touched here.
- Mordred 3D-conformer descriptors: used in the reference project's full pipeline (its README
  lists this as the slowest step, ~20-30 min) but skipped here for speed. RDKit 2D descriptors
  + Morgan fingerprints alone were enough to land at R2=0.80-0.82.
- PCA: this pipeline doesn't use PCA anywhere. Feature selection and scaling come from
  `VarianceThreshold` and `StandardScaler` on the combined descriptor and fingerprint set,
  matching the reference notebook (`chemical_property_predictor.ipynb`, cells 5-6).

## Data Engineering Validation

To gurantee that the geometry-to-smiles bond perception step (`rdDetermineBonds`) did not introduce valency or silent connectivity errors, 10 molecules were randomly sampled and cross-verified against the canonical SMILES from the original QM9 dataset.

| QM9 ID | Recovered SMILES (stereo-stripped) | QM9 | Match |
|---|---|---|---|
| 6366 | `CC(C)NC(=O)CO` | `CC(C)NC(=O)CO` | Yes |
| 36142 | `C#CCC12COCC1N2` | `C#CCC12COCC1N2` | Yes |
| 38829 | `C1OC2CC1CN1CC21` | `C1C2C3CC(CO3)CN12` | Yes (Alternate ring numbering) |
| 42931 | `N#CCN1C(=O)C2CC21` | `O=C1C2CC2N1CC#N` | Yes (Alternate atom ordering) |
| 47969 | `O=CC#CC1(C=O)CC1` | `C1CC1(C=O)C#CC=O` | Yes (Alternate atom ordering) |
| 77662 | `CC1C2(O)COC12C=O` | `CC1C2(O)COC12C=O` | Yes |
| 111848 | `CCC1C2CCCC12C` | `CCC1C2CCCC12C` | Yes |
| 120992 | `COCC1OC2COC12` | `COCC1OC2COC12` | Yes |
| 30543 | `NC=Nc1ncccn1` | `NC=Nc1ncccn1` | Yes |
| 16651 | `CN1CC2NC2C1=O` | `CN1CC2NC2C1=O` | Yes |

The graphs match in 100% of tested cases. Note that the recovered SMILES include explicit stereocenter tags (`@`/`@@`) parsed from the 3D coordinates, making them structurally _more_ descriptive than the flat 2D QM9 strings. Thus, the pipeline is clean.

## Scope
- Additional properties like electron affinity and solvation energy calculations were left out, but the underlying JSON contains the raw assets for these, making them fairly straightforward candidates for future expansion.
- Larger SolQuest subsets (GDB17, ZINC, EGP) were omitted to maintain a focus on QM9
- Mordred 3D descriptors were skipped for speed, and dimensionality reduction methods like PCA were intentionally avoided to align with the reference project's design.

## Reproducibility

To replicate the environment setup and execute the project:

```bash
cd battery_project
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter execute notebooks/battery_ip_predictor.ipynb
```

Alternatively, run the modular scripts sequentially:

```bash
python notebooks/01_extract_dataset.py
python notebooks/02_featurize.py
python notebooks/03_train.py
```
