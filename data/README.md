# Data

`Clean_BangPaKong_2.xlsx` — water-level readings from the Marine Department monitoring station in
Tha Kham Subdistrict, Bang Pakong District, Chachoengsao Province, Thailand.

| Property | Value |
|---|---|
| Records | 353,644 |
| Columns | `DateTime`, `WaterLevel(m)` |
| Period | 7 May 2019 16:48 – 23 June 2026 15:45 |
| Nominal interval | about 10 min (9–16 min in practice) |
| Datum | as supplied by the source; levels are relative, not corrected for vertical land motion |
| Time zone | as supplied (assumed local time) |

The file is the "cleaned" export supplied by the data provider. It still contains artefacts that the
quality-control step of this study removes:

| Issue | Count |
|---|---|
| Duplicate timestamps (merged to their median) | 665 timestamps / 974 extra readings |
| Duplicates disagreeing by more than 0.10 m (dropped) | 1 |
| Zero readings | 38 |
| Stuck readings at 5.67–5.69 m (Nov–Dec 2022) | 563 |
| Flat-line runs of at least three hours | 608 |
| Spikes (> 0.25 m from a 70-min running median) | 248 |

351,212 readings remain after quality control. `results/tables/qc_log.csv` holds the same summary as
produced by the code.

Please cite the Marine Department as the source of the raw measurements when reusing this file.
