<div align="center">

# CARISMA

**Crop Area Recognition and Identification via SAR Mapping in Agriculture**

Winter crop type mapping in eastern Uttar Pradesh from Sentinel-1 radar (Rabi 2021-22), with an interactive story, map and honest accuracy assessment.

[![Live site](https://img.shields.io/badge/Live_site-open-1b3a33?style=for-the-badge&logo=githubpages&logoColor=white)](https://vaibhavnagare-gis.github.io/sar-crop-mapping-eastern-up/)
[![License: MIT](https://img.shields.io/badge/code-MIT-f2c230?style=for-the-badge)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/results-CC_BY_4.0-3f8f3a?style=for-the-badge)](#licence-and-credits)

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![Sentinel-1](https://img.shields.io/badge/Sentinel--1-C--band_SAR-0b3d91?style=flat-square)
![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-data-4285F4?style=flat-square&logo=googleearth&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![MapLibre](https://img.shields.io/badge/MapLibre_GL-web_map-396CB2?style=flat-square&logo=maplibre&logoColor=white)
![Status](https://img.shields.io/badge/status-work_in_progress-d55e00?style=flat-square)

<a href="https://www.linkedin.com/in/vaibhav-nagare-gis"><img src="docs/assets/img/button-linkedin.svg" alt="Vaibhav Nagare on LinkedIn" height="40"></a>
&nbsp;
<a href="https://github.com/VaibhavNagare-GIS"><img src="docs/assets/img/button-github.svg" alt="Vaibhav Nagare on GitHub" height="40"></a>

</div>

---

## What this is
A Random Forest model maps **Wheat, Mustard/Lentil and Sugarcane** from Sentinel-1 backscatter (VV and VH, ascending and descending passes). It is scored on fields and spatial blocks that were never used for training, and shown on a web map where every field can be compared with the survey. Sentinel-2 is used only as a comparison.

**Live story and map:** https://vaibhavnagare-gis.github.io/sar-crop-mapping-eastern-up/

## Results
Frozen settings, scored once. Intervals are 95% field-level bootstrap. Macro-F1 is the headline because accuracy alone is misleading with unequal classes.

| Area | Level | Macro-F1 | Note |
|---|---|---|---|
| Study block (test set) | field vote | 0.560 [0.496, 0.625] | 412 test fields |
| Study block (test set) | pixel map | 0.525 | overall accuracy 0.671, below the always-wheat baseline of 0.675 |
| Nearby area (new) | field vote | 0.764 [0.667, 0.826] | 263 fields, baseline accuracy 0.690 |
| Far area (new) | field vote | 0.772 [0.730, 0.816] | 615 fields, two classes only (no Sugarcane), baseline accuracy 0.564 |

The three areas have different class mixes, so their scores are not directly comparable. Full numbers: `results/final_*.json` and `docs/data/results/`.

Study-block comparison at field level (same test set):

| Model | Macro-F1 | AUC |
|---|---|---|
| Sentinel-1, ascending + descending | 0.560 [0.496, 0.625] | 0.837 |
| Sentinel-1, ascending only | 0.536 [0.471, 0.597] | 0.798 |
| Sentinel-1 + Sentinel-2 | 0.609 [0.545, 0.668] | 0.882 |
| Sentinel-1 + Google satellite embeddings | 0.597 [0.529, 0.657] | 0.860 |

## Limitations
- Accuracy is modest: Mustard/Lentil fields are often labelled Wheat and most Sugarcane fields are missed.
- Small fields mix neighbouring fields in 10 m pixels; accuracy rises with field size.
- Labels come from a survey after harvest and are not a probability sample, so mapped areas are not unbiased area estimates.
- Lentil is merged with Mustard because there are too few Lentil fields.

## Data
- **Labels:** AgriFieldNet field survey (Radiant Earth Foundation and IDinsight 2022, CC-BY-4.0, DOI 10.34911/rdnt.wu92p1).
- **Study block:** six 0.5 degree cells, 454 label chips, 2,522 fields.
- **Radar:** Sentinel-1 GRD from Google Earth Engine, 12-day composites from 1 Oct 2021 to 28 May 2022 (20 windows); features VV, VH and VH minus VV in dB.
- **Validation:** whole fields and whole 0.1 degree blocks are never split between training and test.

## Quick start (Windows, VS Code terminal, repository root)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tests/run_all.py            # must end with ALL SELFTESTS OK
python -m http.server 8000 --directory docs   # preview the site at http://localhost:8000
```
Earth Engine steps need `earthengine authenticate` with the account that owns your Cloud project.

## Rebuild the web data
```powershell
python src/p8_tiles.py                 # add --masked, --conf, --set _near, --set _far for the other layers
python src/p8_geojson.py               # field polygons (also --set _near / _far)
python src/p8_signatures.py --folder data/phase2/2021-10-01_mt2
python tools/check_site.py             # run before every push: all of docs/ must be committed
python tools/push_in_batches.py        # pushes new tile files in small commits (avoids HTTP 408)
```

## Repository layout
```
src/       shared modules and pipeline scripts p0_ to p8_, runbooks run_*.py
tests/     self-tests on synthetic data; python tests/run_all.py
tools/     image helpers, publish check, batched push
results/   small committed outputs: frozen configuration, metrics, confusion matrices
docs/      GitHub Pages site (HTML, CSS, JS, tiles, field polygons, results)
reports/   codebook and project documents
data/      not committed (labels, features, dense chips); regenerated by the scripts
```

## Licence and credits
Code: MIT. Results and text: CC-BY-4.0.
Contains modified Copernicus Sentinel data (2021-2022). Labels: Radiant Earth Foundation and IDinsight (2022). Land cover: ESA WorldCover. Basemaps: &copy; OpenStreetMap contributors; imagery &copy; Esri, Maxar, Earthstar Geographics and the GIS User Community. Web libraries: MapLibre GL JS (BSD-3-Clause), Apache ECharts (Apache-2.0), GSAP (standard no-charge licence), Lenis (MIT). Fonts: Young Serif and Atkinson Hyperlegible Next (SIL OFL 1.1). Illustrations are original artwork made for this project.

To cite, use the "Cite this repository" button (`CITATION.cff`).
