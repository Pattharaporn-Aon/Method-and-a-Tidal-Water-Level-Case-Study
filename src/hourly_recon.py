import numpy as np, pandas as pd, pickle, glob, tide
from experiments import cases
names = list(tide.CONST)
h = pd.read_pickle('hourly.pkl'); ho = h.dropna()
d = pd.read_pickle('daily.pkl')
C = cases()

def to_hourly(daily_series, hidx):
    s = daily_series.copy(); s.index = s.index + pd.Timedelta('12h')
    u = s.index.union(hidx)
    return s.reindex(u).interpolate('time', limit_direction='both').reindex(hidx).values

# ---------------- validation on realistic windows (C cases) ----------------
val_rows = []; examples = {}
for ci in range(12, 22):
    rows, preds = pickle.load(open(f'exp/case_{ci:02d}.pkl','rb'))
    m = C[ci]['mask']; blk_days = d.index[m]
    b0, b1 = blk_days[0], blk_days[-1] + pd.Timedelta('1D')
    inblk = (ho.index >= b0) & (ho.index < b1)
    coef = tide.fit(ho.index[~inblk], ho.values[~inblk], names)          # tide fitted without the window
    ht = ho[inblk]; tid = tide.predict(ht.index, names, coef)
    # training daily means (window removed) + model predictions inside window
    base = d.dmwl.where(~pd.Series(m, d.index))
    rec = {}
    li_h = ho[~inblk].reindex(ho[~inblk].index.union(ht.index)).interpolate('time').reindex(ht.index).values
    rec['LI (hourly)'] = li_h
    rec['Tide + Z0'] = tid + coef[0]
    pidx = pd.DatetimeIndex(preds['date'])
    preds['GP-2M-SA'] = pickle.load(open(f'exp/sa_{ci:02d}.pkl','rb'))[1]
    for lab in ('St','GP','GP-2M','GP-2M-SA'):
        s = base.copy(); s.loc[pidx] = preds[lab]
        s = s.loc[b0-pd.Timedelta('3D'):b1+pd.Timedelta('3D')]
        rec[f'Tide + {lab}'] = tid + to_hourly(s, ht.index)
    for k, v in rec.items():
        e = v - ht.values
        val_rows.append(dict(case=ci, window=C[ci]['label'], method=k, n_hours=len(e), RMSE_cm=100*np.sqrt(np.mean(e**2)),
                             MAE_cm=100*np.mean(np.abs(e)), R2=1-np.sum(e**2)/np.sum((ht.values-ht.values.mean())**2)))
    if ci in (13, 18): examples[ci] = pd.DataFrame({'obs':ht.values, **rec}, index=ht.index)
val = pd.DataFrame(val_rows)
print(val.groupby('method')[['RMSE_cm','MAE_cm','R2']].mean().sort_values('RMSE_cm'))

# ---------------- final hourly reconstruction ----------------
ff = pickle.load(open('final_fit.pkl','rb'))
CHOICE = 'GP2M32SA'
gidx = ff['grid_idx']; n_d = len(d)
L = pd.Series(ff[CHOICE]['mean'][:n_d], d.index); Lsd = pd.Series(ff[CHOICE]['sd_f'][:n_d], d.index)
coef = np.load('tide_coef_all.npy')
tid = tide.predict(h.index, names, coef)
Lh = to_hourly(L, h.index); Lsdh = to_hourly(Lsd, h.index)
model = tid + Lh
resid = pd.Series(h.values - model, h.index)
hf_sd = float(resid.std())
isn = h.isna()
grp = (isn != isn.shift()).cumsum()
glen = isn.groupby(grp).transform('sum')
short = isn & (glen <= 6)
r_int = resid.interpolate('time', limit_area='inside')
rec = pd.Series(np.where(~isn, h.values, np.where(short, model + r_int.values, model)), h.index)
flag = np.where(~isn, 0, np.where(short, 1, 2))
sd = np.where(~isn, 0.0, np.where(short, 0.5*hf_sd, np.sqrt(Lsdh**2 + hf_sd**2)))
hourly_out = pd.DataFrame({'observed_m':h.values, 'tide_m':tid, 'low_freq_level_m':Lh, 'reconstructed_m':rec.values,
                           'sd_m':sd, 'flag':flag}, index=h.index)
hourly_out.index.name='datetime'
pickle.dump(dict(val=val, examples=examples, hourly=hourly_out, hf_sd=hf_sd, choice=CHOICE,
                 tide_table=tide.amplitudes(names, coef), Z0=coef[0]), open('hourly_recon.pkl','wb'))
print('hf_sd', hf_sd, pd.Series(flag).value_counts())
