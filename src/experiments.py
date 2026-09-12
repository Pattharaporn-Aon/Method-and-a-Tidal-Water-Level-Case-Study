import numpy as np, pandas as pd, pickle, sys, time
from models import *
d = pd.read_pickle('daily.pkl'); idx = d.index; t_all = tyears(idx); y_all = d.dmwl.values
obs = ~np.isnan(y_all); n = len(y_all)

def cases():
    C = []
    for pct in (10,20,30,40):
        L = int(round(n*pct/100))
        for cpos in (0.35, 0.65):
            s = int(n*cpos - L/2); m = np.zeros(n,bool); m[s:s+L] = True
            C.append(dict(group='A_interior_block', label=f'{pct}% block centred {int(cpos*100)}%', pct=pct, mask=m))
    for pct in (10,20,30,40):
        L = int(round(n*pct/100)); m = np.zeros(n,bool); m[n-L:] = True
        C.append(dict(group='B_end_prediction', label=f'last {pct}%', pct=pct, mask=m))
    for yr in (2020,2022,2023,2024,2025):
        m = (idx>=f'{yr}-06-25')&(idx<=f'{yr}-09-07')
        C.append(dict(group='C_realistic_JunSep_75d', label=f'{yr}-06-25 to {yr}-09-07', pct=None, mask=np.asarray(m)))
    for yr in (2021,2022,2023,2024,2025):
        m = (idx>=f'{yr}-11-27')&(idx<=f'{yr+1}-01-04')
        C.append(dict(group='C_realistic_NovJan_39d', label=f'{yr}-11-27 to {yr+1}-01-04', pct=None, mask=np.asarray(m)))
    rng = np.random.default_rng(2024)
    for pct in (10,20,30,40):
        m = np.zeros(n,bool)
        while m[obs].mean() < pct/100:
            L = rng.integers(1,11); s = rng.integers(0,n-L); m[s:s+L]=True
        C.append(dict(group='D_scattered_1to10d', label=f'{pct}% scattered 1-10 d gaps', pct=pct, mask=m))
    return C

def metrics(yt, yp, sd=None):
    e = yp-yt; out = dict(RMSE_cm=100*np.sqrt(np.mean(e**2)), MAE_cm=100*np.mean(np.abs(e)), bias_cm=100*np.mean(e))
    if sd is not None: out['cover95_pct'] = 100*np.mean(np.abs(e) <= 1.96*sd)
    return out

def run(ci, c):
    test = c['mask'] & obs; train = obs & ~c['mask']
    tt, yt = t_all[train], y_all[train]; ts, ys = t_all[test], y_all[test]
    rows = []; preds = {'date': idx[test], 'obs': ys}
    # linear interpolation / persistence
    li = np.interp(ts, tt, yt); preds['LI'] = li
    rows.append(dict(model='LI', **metrics(ys, li)))
    st = StModel().fit(tt, yt); p,_,_ = st.predict(ts); preds['St'] = p
    rows.append(dict(model='St', **metrics(ys, p), rate_mm_yr=1000*st.rate()[0]))
    ytr = np.where(train, y_all, np.nan)
    tv = TVSModel().fit(t_all, ytr); p = tv.mean[test]; preds['TVS']=p
    rows.append(dict(model='TVS', **metrics(ys, p, tv.sd[test]), rate_mm_yr=1000*tv.rate()[0]))
    for k, lab in (('GP','GP'),('GP2M32','GP-2M')):
        g = GPModel(k).fit(tt, yt); m, sd, _ = g.predict(ts)
        sdy = np.sqrt(sd**2 + g.params['sn2']); preds[lab] = m; preds[lab+'_sd'] = sdy
        rows.append(dict(model=lab, **metrics(ys, m, sdy), rate_mm_yr=1000*g.rate()[0]))
    for r in rows: r.update(case=ci, group=c['group'], label=c['label'], pct=c['pct'], n_test=int(test.sum()), n_train=int(train.sum()))
    return rows, preds

if __name__ == '__main__':
    C = cases(); ids = [int(a) for a in sys.argv[1:]] or range(len(C))
    for ci in ids:
        tic = time.time(); rows, preds = run(ci, C[ci])
        pickle.dump((rows, preds), open(f'exp/case_{ci:02d}.pkl','wb'))
        print(ci, C[ci]['label'], f'{time.time()-tic:.0f}s', [(r['model'], round(r['RMSE_cm'],2)) for r in rows], flush=True)
