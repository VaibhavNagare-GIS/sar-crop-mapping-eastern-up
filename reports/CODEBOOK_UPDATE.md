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
| `src/p6_objects.py` | Pixel vs true field vs superpixel objects; `--smoke` for a quick real-data check | `data/phase6/objects.csv` |
| `src/p6_context.py` | Local mean, standard deviation and entropy features | `data/phase6/context.csv` |
| `src/p1_orbit_summary.py` | Chips covered by each Sentinel-1 orbit and orbit combination | printed |
| `src/p1_pixels.py`, `p1_coverage.py` | `--set _near` or `--set _far` builds an evaluation set from cells never used for training | `data/phase1<set>/` |
| `src/run_5.py`, `run_6.py`, `run_7.py` | Runbooks: dense chips, accuracy experiments, extra evaluation set | |
| `src/p4_report.py` | Summary tables from `results/` | `results/test_summary.md` |
| `src/p6_densecv.py` | Map-level cross-validation on train chips: pixel, smoothing, grid, superpixel vote, superpixel Random Forest, true-field upper bound | `data/phase6/densecv.csv` |
| `src/p5_chip_features.py`, `p5_dense_table.py` | `--orbit DESC_165` or `ASC_158` downloads and tabulates an extra orbit on the chips that have at least 18 scenes of it | `data/phase5/dense_<orbit>_mt2/`, `data/phase2/dense_<orbit>_mt2/` |
| `src/run_8.py` | Runbook: third orbit, two-orbit against three-orbit on the same pixels | |

## Results of the training-data experiments (pixel level, 3 classes, block-wise CV)
| Experiment | Result |
|---|---|
| Dense features against sampled features | F1 0.590 against 0.587, AUC 0.811 against 0.810 |
| Class weights on probabilities | held-out gain -0.003 (field) and -0.004 (pixel) |
| Local mean, standard deviation and entropy | F1 0.596 against 0.590, AUC 0.815 against 0.811 |
| Voting over objects (labelled pixels only) | pixel 0.589; superpixels of 15, 30, 60 px 0.618, 0.625, 0.629; true fields 0.684 |
