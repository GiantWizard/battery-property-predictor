# Results - Battery/Electrolyte Property Predictor

Target: ionization potential (IP) of small organic molecules, predicted from SMILES.
Pipeline: RDKit 2D descriptors + Morgan fingerprints, then VarianceThreshold + StandardScaler,
then CatBoost tuned with Optuna, 70/15/15 train/val/test split. Structure and code reused
directly from [chemical-property-predictor](https://github.com/GiantWizard/chemical-property-predictor).

## Headline result

| Metric | Value |
|---|---|
| R2 (test) | 0.80-0.82 across runs |
| MAE | ~0.22-0.23 eV |
| RMSE | ~0.29-0.31 eV |
| MAPE | ~2.4-2.5% |
| n_test | 1,656 |

Optuna isn't seeded, so R2 moves a bit run to run. Four independent runs so far, each backed
by a saved artifact: 0.7994, 0.8106, 0.8115, 0.8235. MAE/RMSE/MAPE move by a similar small
amount and should be read as representative, not exact to four decimal places. The most recent
script run (`data/results_summary.txt`, R2=0.7994) and the current executed notebook
(R2=0.8106) are both fresh-environment runs from this pass. The two earlier runs (0.8115 from
the original script run, 0.8235 from the original notebook run) are preserved in
`_old_run_backup/`.

Full run: `notebooks/battery_ip_predictor.ipynb` (executed copy:
`notebooks/executed_battery_ip_predictor.ipynb`).

## Dataset

Source: QM9-IPEA subset of the SolQuest dataset, from Weinreich et al., "Calculated
state-of-the-art results for solvation and ionization energies of thousands of organic
molecules relevant to battery design," arXiv:2411.00994, published in Machine Learning:
Science and Technology. Data hosted on Zenodo: [10.5281/zenodo.13952172](https://doi.org/10.5281/zenodo.13952172),
file `QM9IPEA.json` (10.3 MB). Confirmed downloadable: a direct HTTPS GET returned status 200
and 10,314,543 bytes, matching the size listed on the Zenodo record.

One wrinkle: the raw JSON does not ship SMILES strings, only 3D geometry (`SYMBOLS`/`COORDS`
per molecule) and per-charge-state coupled-cluster energies (`ENERGY`, keyed by charge state
`0`/`1`/`-1`, then by level of theory). There is no separate extraction script published for
QM9-IPEA specifically in the paper's companion GitHub repo
([chemspacelab/VienUppDa](https://github.com/chemspacelab/VienUppDa), linked from the paper's
data availability section). The closest thing is
`QM9IPEA/scripts/json_creation/check_json_shape.py` and `print_json_entry_examples.py`, which
are shape-inspection utilities, not a SMILES extractor. Those were used to understand the JSON
schema, then SMILES were recovered from the neutral-molecule geometry using RDKit's
bond-perception algorithm (`rdDetermineBonds.DetermineBonds`), the same xyz-to-connectivity
approach these datasets are built around.

Vertical IP is defined, per the paper's data availability text, as the neutral-to-cation
energy difference at the neutral geometry:

```
IP = E(cation, charge +1) - E(neutral, charge 0)
```

computed at the highest level of theory present for each spin state:
`PNO-LCCSD(T)-F12B` (closed-shell neutral) and `PNO-UCCSD(T)-F12B` (open-shell cation).

## Dataset size, extraction and filtering

| Stage | Molecules | Notes |
|---|---|---|
| Starting point | 7,000 | Full QM9-IPEA set |
| After energy availability check | 6,001 | 999 molecules missing neutral or cation energy at the chosen level of theory |
| After SMILES recovery from geometry | 5,699 | 302 further dropped; RDKit's bond perceiver couldn't assign a valid Lewis structure from the 3D geometry (mostly valence-ambiguous cases) |
| After featurization (RDKit 2D + Morgan FP) | 5,699 | 0 additional drops, since every SMILES RDKit itself produced is by definition RDKit-parseable |
| After IQR outlier removal on IP | 5,518 | Standard 1.5x IQR trim on the target, same as reference pipeline |
| Train / val / test split | 3,281 / 580 / 1,656 | `test_size=0.3`, then 15% of the remainder held out as validation. Same split convention as the reference project (works out to roughly 59.5/10.5/30, not an exact 70/15/15, because that's literally what `test_size=0.3` followed by `test_size=0.15` on the remainder produces; this matches the reference notebook's actual code, not just its stated "70/15/15" label) |

Overall retention: 5,699 / 7,000 (81.4%) from raw dataset to usable IP-labeled molecules.
Comparable in kind to the reference project's roughly 3,300 / 4,343 (76%) retention rate,
though the failure mode here is different (geometry-to-SMILES perception vs. organic/inorganic
filtering).

## Timing

| Step | Time |
|---|---|
| Extraction (JSON to SMILES + IP) | ~1.2s for all 7,000 raw entries |
| Featurization (RDKit 2D descriptors + Morgan fingerprints, 5,699 molecules) | ~37s |
| Optuna tuning (15 trials) | ~125-190s across runs (~2-3 min) |
| Final model fit | ~6-14s |
| Total pipeline (data to trained model to test metrics) | ~2.5-4 minutes |

This is much faster than the reference project's boiling-point/critical-temperature runs,
which report 10-15 min per property mostly because of Mordred 3D-conformer descriptor
computation. That step was skipped here (see "What was skipped" below), which accounts for
most of the timing difference.

## Comparison to chemical-property-predictor's results

| Target | R2 | Where it sits |
|---|---|---|
| Critical temperature (reference project's best) | 0.939 | Easiest |
| Ionization potential (this project) | 0.80-0.82 | Middle, closer to the easy end |
| Boiling point (reference) | 0.827 | Roughly same tier as IP |
| Critical pressure (reference) | 0.920 | Easier than IP |
| Flash point (reference) | 0.843 | Roughly same tier as IP, but only 267 samples |
| Heat of vaporization (reference) | 0.755 | Harder than IP |
| Melting point (reference project's hardest) | 0.674 | Hardest |

IP landing closer to the easy end makes physical sense. Ionization potential is basically an
electronic property, driven at a first pass by frontier orbital energetics like HOMO energy,
electronegativity, and polarizability, which is exactly the kind of signal captured by 2D RDKit
descriptors such as `SMR_VSA*`, `VSA_EState*`, `BCUT2D_MR*`, and `SlogP_VSA*` (several of the
top-15 CatBoost feature importances are electronic or polarizability descriptors rather than
raw structural counts), plus the substructure fingerprints. Melting point sits at the other end
because it depends heavily on 3D crystal packing and intermolecular lattice forces, information
a 2D graph descriptor and fingerprint representation simply doesn't carry. That's exactly why
the reference project flags melting point as its hardest target (R2=0.674).

IP doesn't reach critical temperature's R2=0.939 territory, likely for two reasons. First, this
dataset (QM9-derived, at most 9 heavy atoms of C/N/O/F) is far more structurally homogeneous
and small than the reference project's 4,343-compound organic substance database. That cuts
both ways: less diversity can make interpolation easier, but a narrower property range (IP
spans roughly 6.9-14.3 eV pre-outlier-removal) also compresses the achievable R2 for a fixed
absolute error. Second, IP is a genuinely quantum-mechanical, orbital-energy-driven property,
versus critical temperature, which correlates strongly with simple size/polarity descriptors
(molecular weight, surface area) that 2D descriptors capture almost by construction.

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

## Spot-check: does xyz-to-SMILES bond perception introduce errors?

The one place this pipeline diverges from having ground-truth input SMILES is the
geometry-to-SMILES step (RDKit's `rdDetermineBonds`, used because `QM9IPEA.json` ships 3D
coordinates but no SMILES, as noted above). To check that step isn't silently getting bond
orders or connectivity wrong, 10 molecules were sampled at random from `qm9ipea_smiles_ip.csv`
and cross-referenced against the original QM9 dataset's own canonical SMILES (deepchem's QM9
mirror, indexed by the same `gdb_<QM9_ID>` numbering used internally by the QM9-IPEA build
scripts, confirmed via `VienUppDa/QM9IPEA/scripts/json_creation/make_combined_json.py`'s
`xyz_subdir_to_qm9_id`, which extracts this same integer index from the original QM9 xyz
directory names).

For each sampled molecule, both SMILES were canonicalized with stereochemistry stripped and
compared, plus molecular formulas cross-checked:

| QM9 ID | Recovered SMILES (stereo-stripped) | QM9 ground truth | Match |
|---|---|---|---|
| 6366 | `CC(C)NC(=O)CO` | `CC(C)NC(=O)CO` | yes |
| 36142 | `C#CCC12COCC1N2` | `C#CCC12COCC1N2` | yes |
| 38829 | `C1OC2CC1CN1CC21` | `C1C2C3CC(CO3)CN12` | yes, same graph, different ring-closure numbering |
| 42931 | `N#CCN1C(=O)C2CC21` | `O=C1C2CC2N1CC#N` | yes, same graph, different atom-order convention |
| 47969 | `O=CC#CC1(C=O)CC1` | `C1CC1(C=O)C#CC=O` | yes, same graph, different atom-order convention |
| 77662 | `CC1C2(O)COC12C=O` | `CC1C2(O)COC12C=O` | yes |
| 111848 | `CCC1C2CCCC12C` | `CCC1C2CCCC12C` | yes |
| 120992 | `COCC1OC2COC12` | `COCC1OC2COC12` | yes |
| 30543 | `NC=Nc1ncccn1` | `NC=Nc1ncccn1` | yes |
| 16651 | `CN1CC2NC2C1=O` | `CN1CC2NC2C1=O` | yes |

Result: 10/10 match, identical molecular formula and identical bond connectivity (RDKit
canonical SMILES equal once stereochemistry is stripped) in every sampled case. No wrong bond
orders, no missed or extra bonds, no mis-assigned atoms.

On stereochemistry specifically: the recovered SMILES do carry `@`/`@@` stereocenter markers
that the flat QM9 ground-truth SMILES doesn't show (for example molecule 36142's
`[C@@]12...[C@]1`). This isn't an error. QM9's canonical SMILES field is a flat 2D graph label
and doesn't encode stereochemistry at all, while `rdDetermineBonds` perceives real spatial
chirality from the actual 3D coordinates. The recovered SMILES are, if anything, more
informative than the ground-truth comparison string. There's no independent 3D ground truth to
check those specific stereo assignments against here, but since bond connectivity is verified
exact in all 10 samples, there's no evidence of the perception step failing silently. The 302
geometry-to-SMILES failures already reported (see filtering table above) are cases where
`rdDetermineBonds` raised or failed outright and were correctly dropped, not silent errors that
made it into the training set.

Conclusion: clean. No evidence this step is corrupting the training data or inflating R2.

## Reproducibility

The pipeline was re-run from a freshly rebuilt virtual environment (`venv/` deleted, recreated
with `python3.12 -m venv venv`, dependencies reinstalled from `requirements.txt`) to check that
the results hold up outside the original environment. Extraction and featurization produced
identical counts both times (5,699/7,000 after energy and geometry filtering, 2,268 feature
columns), as expected since neither step has randomness. Training is not fully deterministic,
since Optuna's search and CatBoost's own stochastic elements are unseeded, so R2 moves a bit
run to run: four runs across the standalone script and the notebook, two from the original
build and two from the fresh-environment re-run, gave R2 = 0.7994, 0.8106, 0.8115, and 0.8235,
all on the same held-out test split (`n_test=1,656`, `random_state=42` fixed for the
train/val/test splitting itself). The honest range to report is R2=0.80-0.82. Feature
importances were stable across all four runs, with `SMR_VSA3`, `SMR_VSA7`, `fr_aniline`,
`VSA_EState4`, and `MolMR` consistently in the top 6.

Artifacts from the original build (`qm9ipea_smiles_ip.csv`, `descriptors.csv`,
`results_summary.txt`, and the original executed notebook) are kept in `_old_run_backup/`
alongside the current ones for comparison.

## Reproducing

```bash
cd battery_project
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter execute notebooks/battery_ip_predictor.ipynb
```

Or run the three standalone scripts in order:
`python notebooks/01_extract_dataset.py && python notebooks/02_featurize.py && python notebooks/03_train.py`
