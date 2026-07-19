# Battery / Electrolyte Property Predictor

Predicts **ionization potential (IP)** of small organic molecules from SMILES. Built by reusing the pipeline from [chemical-property-predictor](https://github.com/GiantWizard/chemical-property-predictor): RDKit descriptors + Morgan fingerprints, CatBoost tuned with Optuna, 70/15/15 split.

See [RESULTS.md](RESULTS.md) for the full writeup, actual R2 (0.80-0.82 across runs), timing,
and comparison to the reference project.

## Layout

```
battery_project/
├── data/
│   ├── QM9IPEA.json              # raw dataset
│   ├── qm9ipea_smiles_ip.csv     # extracted SMILES + IP
│   ├── descriptors.csv           # featurized dataset
│   └── results_summary.txt       # metrics from the standalone-script run
├── notebooks/
│   ├── battery_ip_predictor.ipynb          # main notebook, all 3 phases
│   ├── executed_battery_ip_predictor.ipynb # executed copy with real outputs
│   ├── 01_extract_dataset.py               # JSON -> SMILES + IP
│   ├── 02_featurize.py                     # RDKit descriptors + fingerprints
│   └── 03_train.py                         # CatBoost + Optuna + eval
├── _old_run_backup/               # from the original build's runs
├── requirements.txt
└── RESULTS.md
```

## Setup

```bash
python3.12 -m venv venv        # RDKit needs 3.12/3.13
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter execute notebooks/battery_ip_predictor.ipynb
```

or the standalone scripts:

```bash
python notebooks/01_extract_dataset.py
python notebooks/02_featurize.py
python notebooks/03_train.py
```

## Dataset

QM9-IPEA subset of SolQuest (Weinreich et al., arXiv:2411.00994, Machine Learning: Science and Technology). Zenodo: [10.5281/zenodo.13952172](https://doi.org/10.5281/zenodo.13952172), file `QM9IPEA.json`. Contains about 7,000 QM9 molecules with coupled-cluster-level ionization potentials and electron affinities. This project uses IP only; EA and multi-solvent solvation energy are not built here.

## Not in scope

- Electron affinity, solvation energy 
- GDB17 / ZINC / EGP portions of SolQuest
- Mordred 3D-conformer descriptors (skipped for speed; RDKit 2D + fingerprints alone hit R2=0.80-0.82). Similar rationale to [chemical-property-predictor](https://github.com/GiantWizard/chemical-property-predictor)
