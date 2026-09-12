# Source modules

The notebook in `notebooks/` runs the whole analysis in one place. These modules are the same
pipeline split into steps, in the order they were used for the paper.

| Order | File | What it does | Output |
|---|---|---|---|
| 1 | `prep.py` | quality control of the 10-min record, hourly series | `clean_10min.pkl`, `hourly.pkl`, `qc_log.csv` |
| 2 | `tide.py` | tidal constituents, nodal corrections, harmonic fit (library) | – |
| 2 | `daily.py` | harmonic fit and tide-filtered daily mean water level | `daily.pkl`, `daily_gaps.csv` |
| 3 | `models.py` | Standard, Bennett-type Kalman (TVS) and GP models (library) | – |
| 4 | `experiments.py` | 26 withheld-data cases for LI, St, TVS, GP, GP-2M | `exp/case_XX.pkl` |
| 4 | `exp_sa.py` | the same cases for GP-2M-SA | `exp/sa_XX.pkl` |
| 5 | `final_fit.py` | full-record fits of every model, forecast to end-2026 | `final_fit.pkl` |
| 5 | `loo.py` | closed-form leave-one-out residuals | `loo.pkl` |
| 6 | `hourly_recon.py` | hourly validation windows and the final hourly reconstruction | `hourly_recon.pkl` |
| 7 | `figures.py`, `paper_figs.py` | figures | `fig/*.png` |
| 7 | `build_xlsx.py` | the results workbook | `BangPakong_GP_Reconstruction.xlsx` |

Input: the workbook in `data/`, read once into a pickle (`pd.read_excel(...).to_pickle('bpk.pkl')`).

Requirements: Python 3.10+, `numpy`, `scipy`, `pandas`, `matplotlib`, `openpyxl`.

Runtime: each GP maximum-likelihood fit on about 2,400 daily values takes 0.5–2 min on two CPU
cores; the 26-case experiment set takes roughly an hour.
