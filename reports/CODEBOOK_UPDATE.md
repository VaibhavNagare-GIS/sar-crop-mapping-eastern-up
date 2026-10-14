# Codebook update (phases 2c to 5a). All code now lives in `src/`; run everything from the repository root as `python src/<script> ...`.
| File | Role | Output |
|---|---|---|
| `src/p2_extra.py` | `angle` (mean incidence angle per pixel, `--pass asc|desc`) and `embed` (Google satellite embeddings 2021 and 2022, benchmark only) | `data/phase2/angle_<pass>.csv`, `embed.csv` |
| `src/p3_feat.py` | Field-level feature ladder on train fields: base, lin, inner, std, angle, norm, emb. Rule: keep a rung if macro-F1 rises by 0.02 or more and AUC does not fall | `data/phase3/feat.csv` |
| `src/p3_windows.py` | Which time windows carry the signal (label-season controls, earliest reliable date). Raw features only | `data/phase3/windows.csv` |
| `src/p4_final.py` | `freeze` writes `results/FROZEN_CONFIG.json`; `test` is the single test evaluation with field-bootstrap intervals and a lock file | `results/` |
| `src/p5_chip_features.py` | `parity` compares dense `computePixels` features with the sampled training features on 3 chips; `download` saves all chips as int16 | `data/phase5/dense_<pass>_mt<k>/<chip>.npy` |
| `src/run_3f.py`, `run_3g.py` | Runbooks (3F: extractions and feature ladder; 3G: window controls and leaf choice) | |
| `tests/run_all.py` | Runs every self-test; each must print SELFTEST OK | |
Notes: window i of folder `<start>` covers `<start> + 12 i` days (end exclusive; the last window of 2021-10-01 ends 28 May 2022 inclusive). `p3_sweep.load(spec, splits)` returns train rows unless a caller asks for test; only `p4_final.py` does.

## Added later
| File | Role | Output |
|---|---|---|
| `src/p5_chip_features.py` | Dense Sentinel-1 stacks per chip (`parity`, `download`, option `--set`) | `data/phase5<set>/dense_<pass>_mt<k>/<chip>.npy` |
| `src/p5_dense_table.py` | Training table taken from the dense stacks | `data/phase2/dense_<pass>_mt<k><set>/features.csv` |
| `src/p6_weights.py` | Class weights on probabilities, scored on held-out blocks | `data/phase3/weights_<level>.json` |
| `src/p6_objects.py` | Superseded by `p6_densecv.py`; kept for reference | `data/phase6/objects.csv` |
| `src/p6_context.py` | Local mean, standard deviation and entropy features | `data/phase6/context.csv` |
| `src/p1_orbit_summary.py` | Chips covered by each Sentinel-1 orbit and orbit combination | printed |
| `src/p1_pixels.py`, `p1_coverage.py` | `--set _near` or `--set _far` builds an evaluation set from cells never used for training | `data/phase1<set>/` |
| `src/run_5.py`, `run_6.py`, `run_7.py` | Runbooks: dense chips, accuracy experiments, extra evaluation set | |
| `src/p4_report.py` | Summary tables from `results/` | `results/test_summary.md` |
| `src/p6_densecv.py` | Map-level cross-validation on train chips: pixel, smoothing, grid, superpixel vote, superpixel Random Forest, true-field upper bound | `data/phase6/densecv.csv` |
| `src/p5_chip_features.py`, `p5_dense_table.py` | `--orbit DESC_165` or `ASC_158` downloads and tabulates an extra orbit on the chips that have at least 18 scenes of it | `data/phase5/dense_<orbit>_mt2/`, `data/phase2/dense_<orbit>_mt2/` |
| `src/run_8.py` | Runbook: third orbit, two-orbit against three-orbit on the same pixels | |
| `src/p6_extra_train.py` | Adds labelled pixels of the `_far` set to every training fold and scores only study blocks | `data/phase6/extra_train.csv` |

| `src/nnprep.py`, `src/p7_nn.py` | TempCNN and a small U-Net on the dense chips (GPU machine; folds and scoring as in `p6_densecv.py`) | `data/phase7/nn.csv` |
| `src/p7_map.py` | Final Random Forest, 5 x 5 smoothed class and confidence GeoTIFFs per chip, probabilities at labelled pixels | `data/phase7/map<set>/`, `labelled<set>.csv`, `results/FINAL_CONFIG.json` |
| `src/p7_eval.py` | Scores the final map with field-level bootstrap intervals (study test fields as a second look, `_near`, `_far`) | `results/final_<name>.json`, confusion CSVs |

## Results of the training-data experiments (pixel level, 3 classes, block-wise CV)
| Experiment | Result |
|---|---|
| Dense features against sampled features | F1 0.590 against 0.587, AUC 0.811 against 0.810 |
| Class weights on probabilities | held-out gain -0.003 (field) and -0.004 (pixel) |
| Local mean, standard deviation and entropy | F1 0.596 against 0.590, AUC 0.815 against 0.811 |
| Voting over objects (labelled pixels only, `p6_objects.py`) | pixel 0.589 to superpixels 0.618 to 0.629. Superseded: the vote ignored unlabelled pixels and the superpixels were square tiles |
| Map-level CV, 367 chips (`p6_densecv.py`) | pixel 0.584 (AUC 0.812); 3, 5, 7 pixel mean filter 0.602, 0.609, 0.608; 8 x 8 blocks 0.549; true-field vote 0.678 |
| Superpixels from the Sentinel-2 snapshot (`p6_densecv.py`) | variance explained beyond a grid 0.42 (25 px, compactness 0.1) and 0.05 (compactness 0.5, a plain grid); best F1 0.580, below the 5 x 5 mean filter (0.609) |
| Far-region pixels as extra training (`p6_extra_train.py`) | base 0.590; far added 0.576; far at weight 0.3 0.593 (AUC 0.811); not adopted |
| Pre-sowing and post-survey windows added to the Rabi windows, field level, ascending only (`p3_sweep.py`) | 0.586 to 0.595, 0.599, 0.599 (AUC 0.811 to 0.820); not adopted |
| Third orbit (`run_8.py`) | DESC_165: 0.606 to 0.620 (222 chips); ASC_158: 0.544 to 0.547 (163 chips); not adopted |
