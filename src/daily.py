import pandas as pd, numpy as np, tide
h = pd.read_pickle('hourly.pkl'); ho = h.dropna()
names = list(tide.CONST)
coef = tide.fit(ho.index, ho.values, names)
np.save('tide_coef_all.npy', coef)
res = ho - tide.predict(ho.index, names, coef, with_mean=True)
cnt = res.resample('D').count()
full = pd.date_range(h.index[0].normalize(), h.index[-1].normalize(), freq='D')
dm = (res.resample('D').mean() + coef[0]).reindex(full)
cnt = cnt.reindex(full).fillna(0).astype(int)
daily = pd.DataFrame({'dmwl': dm.where(cnt>=18), 'hours': cnt})
daily.index.name='date'
daily.to_pickle('daily.pkl')
print(daily.dmwl.describe(), daily.dmwl.isna().sum())
m = daily.dmwl.isna()
blk = (m != m.shift()).cumsum()[m]
gaps = daily[m].groupby(blk).apply(lambda d: pd.Series({'start':d.index[0].date(),'end':d.index[-1].date(),'days':len(d)}))
gaps = gaps.sort_values('days', ascending=False).reset_index(drop=True)
gaps.to_csv('daily_gaps.csv', index=False); print(gaps.to_string())
