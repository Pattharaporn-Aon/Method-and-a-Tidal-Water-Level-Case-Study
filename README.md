[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pattharaporn-Aon/Method-and-a-Tidal-Water-Level-Case-Study/blob/main/notebooks/Tidal_Water_Level_Case_Study.ipynb)

# Gaussian Process Regression for Reconstructing Missing Values in Sensor Time Series

Data, code and results for the tidal water-level case study at Bang Pakong, Chachoengsao, Thailand
(Marine Department monitoring station, May 2019 – June 2026).

The study applies Gaussian process (GP) regression with an explicit mean function — the set-up of
Xu et al. (2024, *GPS Solutions* 28:79) — to a seven-year tide-gauge record, and compares it with a
standard harmonic model, a Kalman model with time-varying seasonal terms, and linear interpolation.

## What is here

| Folder | Contents |
|---|---|
| `data/` | `Clean_BangPaKong_2.xlsx` — 353,644 water-level readings at ~10-min spacing (see `data/README.md`) |
| `notebooks/` | `Tidal_Water_Level_Case_Study.ipynb` — the full analysis in one Colab notebook, with outputs |
| `src/` | the same pipeline as separate Python modules (see `src/README.md`) |
| `results/figures/` | figures produced by the analysis |
| `results/tables/` | quality-control log, tidal constituents, model fits, gap experiments, hourly validation |
| `results/BangPakong_GP_Reconstruction.xlsx` | reconstructed daily and hourly series with 95 % intervals |

## Method in brief

1. **Quality control** of the 10-min record: duplicate timestamps merged, zeros, a stuck 5.67–5.69 m
   plateau, flat-line runs of at least three hours and spikes (> 0.25 m from a 70-min running median)
   removed — 1,458 readings in total.
2. **Hourly series** by interpolation between readings no more than 30 min apart (6.4 % of hours missing).
3. **Tidal harmonic analysis** with 33 constituents and nodal corrections; tides explain 94.7 % of the
   hourly variance (K1 0.65 m, M2 0.56 m, O1 0.42 m, S2 0.27 m; form factor 1.28).
4. **Daily mean water level (DMWL)**: daily mean of the non-tidal residual plus Z0, for days with at
   least 18 valid hours — 2,412 observed and 193 missing days.
5. **Models for the DMWL**
   * `St` — trend + constant annual and semiannual harmonics
   * `TVS` — Bennett-type model with random-walk seasonal coefficients and AR(1) noise (Kalman + RTS smoother)
   * `GP` — mean h(t) = [1, t, sin 2πt, cos 2πt] with a Matérn 3/2 kernel (Xu et al. 2024)
   * `GP-2M-SA` — extension with two Matérn 3/2 components and a semiannual term in the basis
   All hyperparameters are estimated by maximising the marginal likelihood with analytic gradients.
6. **Evaluation** by leave-one-out residuals and by 26 withheld-data experiments (contiguous interior
   blocks, end-of-record prediction, realistic Jun–Sep and Nov–Jan windows, scattered 1–10-day gaps).
   Every model is refitted on the remaining data in each case.

## Main results

**Full record (2,412 daily values)**

| Model | Rate (mm/yr) | LOO RMSE (cm) | AIC |
|---|---|---|---|
| St (Gauss–Markov noise) | 2.46 ± 1.81 | 9.56 | −5441.3 |
| TVS (Kalman) | 2.46 ± 1.85 | – | – |
| GP, Matérn 3/2 (Xu et al.) | 2.59 ± 1.75 | 6.83 | −5385.6 |
| GP, Matérn 1/2 | 2.58 ± 1.97 | 6.81 | −5415.6 |
| GP, Matérn 3/2 + quasi-periodic | 2.49 ± 1.60 | 6.77 | −5417.0 |
| GP-2M-SA | 2.44 ± 2.20 | **6.67** | **−5480.9** |

**Mean RMSE (cm) of withheld daily values**

| Group | LI | St | TVS | GP | GP-2M-SA |
|---|---|---|---|---|---|
| A: interior block (8 cases) | 16.10 | 9.86 | 9.79 | 10.00 | **9.76** |
| B: end of record (4) | 19.89 | **9.98** | 10.06 | 10.28 | 9.99 |
| C1: Jun–Sep, 75 d (5) | 9.21 | 8.62 | 8.65 | **8.31** | 8.59 |
| C2: Nov–Jan, 39 d (5) | 12.92 | 9.59 | **9.53** | 10.01 | 9.54 |
| D: scattered 1–10 d (4) | 10.16 | 9.81 | 9.08 | 9.25 | **8.96** |
| All 26 cases | 13.83 | 9.58 | 9.45 | 9.60 | **9.40** |

The 95 % predictive intervals of GP-2M-SA covered 93–98 % of the withheld values. Hourly
reconstruction (tidal prediction + GP daily level) reached an RMSE of 14.9 cm (R² = 0.965) over ten
withheld windows, against 114 cm for hourly linear interpolation.

Two findings differ from Xu et al. (2024). The large in-sample RMSE reduction they report is not
reproduced out of sample: a GP with a small noise variance interpolates the training data, and its
in-sample error says little about gap filling. In addition, no time-varying seasonal signal was
detected here — the maximum-likelihood random-walk variances went to zero and the decay length of the
quasi-periodic kernel ran to its bound.

## How to run

**Notebook (recommended).** Open `notebooks/Tidal_Water_Level_Case_Study.ipynb` in Colab or Jupyter,
put `Clean_BangPaKong_2.xlsx` in the working folder, and run all cells. The gap experiments take
about an hour on a standard Colab CPU; every other step takes a few minutes.

**Modules.** See `src/README.md` for the run order. Requirements: Python 3.10+, `numpy`, `scipy`,
`pandas`, `matplotlib`, `openpyxl`.

## Reference

K. Xu, S. Hu, S. Jin, J. Li, W. Zheng, J. Wang, Y. Zhu, K. Li, A. Ren and Y. Liu,
"Reconstruction of geodetic time series with missing data and time-varying seasonal signals using
Gaussian process for machine learning," *GPS Solutions*, vol. 28, Art. no. 79, 2024.

## License

Code in this repository is released under the MIT License. The water-level
measurements in `data/` are the property of the Marine Department of Thailand
and are redistributed here for reproducibility; please credit the Marine
Department when reusing them.
