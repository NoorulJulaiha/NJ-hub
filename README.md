# Subject-Independent Evaluation of Multimodal Wearable Sensing and Deep Learning for Sports Injury-Risk Prediction

Code, results and figures for the article:

> [Authors]. *Subject-Independent Evaluation of Multimodal Wearable Sensing and Deep Learning for Sports Injury-Risk Prediction.* [Journal], 2026. [DOI after acceptance]

## What is in this repository

```
code/        Python scripts for the three studies
notebooks/   Google Colab notebooks (run in the browser, no installation)
data/        Study 1 data + instructions for downloading the public datasets
results/     All result tables (CSV) reported in the paper
figures/     All figures (300 dpi PNG)
```

| Study | Data | Script | Validation |
|---|---|---|---|
| 1 – Bench test | Simulated 13-channel device stream (100 Hz) | `code/run_experiments.py` | Segment-blocked, purged 5-fold CV vs random split |
| 2 – WESAD | Public chest + wrist dataset, 15 subjects | `code/wesad_pipeline.py`, `notebooks/WESAD_Validation_Colab_v2.ipynb` (recommended; saves to Drive and resumes) | Leave-one-subject-out vs random split |
| 3 – Runners | Public 7-year training logs, 74 athletes, 583 injuries | `code/run_runners.py`, `code/finish_runners.py`, `code/runner_extra.py` | Athlete-grouped CV, original 64/10 hold-out, random split |

## How to reproduce

**Easiest – Google Colab:** open a notebook from `notebooks/`, choose *Runtime → Run all*.

**Local:**
```bash
pip install -r requirements.txt
cd code
python run_experiments.py --data ../data --out ../output/study1          # Study 1 (~25 min CPU)
python run_runners.py     --data ../data/runners --out ../output/study3  # Study 3 (~1 h CPU)
python finish_runners.py  --data ../data/runners --out ../output/study3  # SHAP + figures
python runner_extra.py                                                   # feature-group ablation, calibration
python wesad_pipeline.py  --root /path/to/WESAD --out ../output/study2    # Study 2 (GPU recommended)
```
Random seeds are fixed (0, 1, 2); results may differ in the third decimal across library versions. WESAD results in `results/study2_wesad` come from one Colab CPU run (seed 0, 10 CNN-GRU epochs).

## Key results

| Study | Leaky random split | Leakage-aware validation |
|---|---|---|
| 1 – Phase recognition (macro-F1) | 0.982–0.988 | 0.781–0.852 |
| 2 – WESAD 3-class, XGBoost (macro-F1) | 1.000 | 0.843 |
| 2 – WESAD, device-matched sensors (macro-F1, 3-class / binary) | – | 0.804 / 0.931 |
| 3 – Injury prediction, XGBoost (AUROC, day) | 0.674 | 0.609 |
| 3 – Bagged XGBoost / GRU (AUROC, athlete-grouped) | – | 0.631 / 0.636 |

## Citation
See `CITATION.cff`. Please also cite the original datasets (Schmidt et al., 2018; Lövdal et al., 2021).

## Licence
Code: MIT. Datasets keep their original licences (see `data/README.md`).
