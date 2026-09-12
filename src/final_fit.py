import numpy as np, pandas as pd, pickle, time
from models import *
d = pd.read_pickle('daily.pkl'); idx = d.index; t_all = tyears(idx); y_all = d.dmwl.values
obs = ~np.isnan(y_all); t, y = t_all[obs], y_all[obs]
fc_idx = pd.date_range(idx[-1]+pd.Timedelta('1D'), '2026-12-31', freq='D'); t_fc = tyears(fc_idx)
grid_idx = idx.append(fc_idx); t_grid = np.r_[t_all, t_fc]
out = {'grid_idx':grid_idx, 'n_obs':int(obs.sum())}
def acf(x, L=30):
    x = x-np.mean(x); return np.array([np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,L+1)])
rows=[]
# --- St
st = StModel().fit(t,y); stgm = GPModel('StGM').fit(t,y)
m,_,_ = st.predict(t_grid); out['St']=dict(mean=m)
res = y - st.predict(t)[0]
rows.append(dict(model='St (constant annual+semiannual)', rate_mm_yr=1000*stgm.rate()[0], rate_sd_mm_yr=1000*stgm.rate()[1],
                 rate_sd_whitenoise_mm_yr=1000*st.rate()[1], model_RMSE_cm=100*np.sqrt(np.mean(res**2)), acf1=acf(res)[0],
                 nll=stgm.nll, AIC=stgm.aic()))
out['St']['resid']=res; out['St']['beta']=st.beta; out['StGM_params']=stgm.params
# --- TVS
tv = TVSModel().fit(t_all, y_all)
res = y - tv.det[obs]
rows.append(dict(model='Bennett-type TVS (Kalman, RW seasonal + AR1)', rate_mm_yr=1000*tv.rate()[0], rate_sd_mm_yr=1000*tv.rate()[1],
                 model_RMSE_cm=100*np.sqrt(np.mean((y-tv.mean[obs])**2)), model_RMSE_det_cm=100*np.sqrt(np.mean(res**2)), acf1=acf(y-tv.mean[obs])[0], nll=tv.nll))
out['TVS']=dict(mean=tv.mean, det=tv.det, sd=tv.sd, params=tv.params)
# --- GP family
kern = {}
for k, lab in (('GP','GP (Matern 3/2; Xu et al. 2024)'),('GP2M32','GP-2M (two-scale Matern 3/2)'),('GPM12','GP (Matern 1/2)'),('GPQP','GP (Matern 3/2 + quasi-periodic)'),('GP2M32SA','GP-2M-SA (two-scale Matern 3/2, annual+semiannual mean)')):
    tic=time.time(); g = GPModel(k).fit(t,y); m, sd, mu = g.predict(t_grid)
    fit_tr = g.predict(t)[0]; res = y - fit_tr
    kern[k] = g
    out[k] = dict(mean=m, sd=np.sqrt(sd**2+g.params['sn2']), sd_f=sd, meanfn=mu, params=g.params, beta=g.beta, resid=res)
    rows.append(dict(model=lab, rate_mm_yr=1000*g.rate()[0], rate_sd_mm_yr=1000*g.rate()[1], model_RMSE_cm=100*np.sqrt(np.mean(res**2)),
                     acf1=acf(res)[0], nll=g.nll, AIC=g.aic(), n_hyper=len(g.lp), **{f'hp_{a}':b for a,b in g.params.items()}))
    print(lab, time.time()-tic, flush=True)
out['table']=pd.DataFrame(rows)
out['acf']={k:acf(out[k]['resid'],60) for k in ('St','GP','GP2M32')}
out['acf']['TVS']=acf(y-tv.mean[obs],60)
pickle.dump(out, open('final_fit.pkl','wb'))
print(out['table'].T)
