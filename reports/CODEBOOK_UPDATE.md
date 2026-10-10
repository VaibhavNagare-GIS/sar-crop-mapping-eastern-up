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
