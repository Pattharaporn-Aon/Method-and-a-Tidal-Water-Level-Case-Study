import numpy as np, pandas as pd, pickle
from models import *
ff = pickle.load(open('final_fit.pkl','rb'))
d = pd.read_pickle('daily.pkl'); t_all = tyears(d.index); y_all = d.dmwl.values
obs = ~np.isnan(y_all); t, y = t_all[obs], y_all[obs]
R = np.abs(t[:,None]-t[None,:]); loo = {}
H = basis(t,(1,2)); hat = H@np.linalg.solve(H.T@H, H.T); e = y - hat@y
res_loo={}; res_loo['St']=e/(1-np.diag(hat)); loo['St'] = 100*np.sqrt(np.mean((e/(1-np.diag(hat)))**2))
for k in ['GP','GP2M32','GPM12','GPQP','GP2M32SA']:
    kf, pn, p0, harm = KERNELS[k]; p = ff[k]['params']
    lp = np.log([p[n] for n in pn+['sn2']])
    K,_ = kf(R, lp[:-1]); K = K + np.eye(len(t))*(p['sn2']+1e-8)
    Hb = basis(t, harm); S = K + Hb@(1e4*np.eye(Hb.shape[1]))@Hb.T
    Si = np.linalg.inv(S); r = (Si@y)/np.diag(Si)
    loo[k] = 100*np.sqrt(np.mean(r**2)); res_loo[k]=r
print(loo)
def acf(x, L=60):
    x=x-x.mean(); return np.array([np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,L+1)])
pickle.dump(dict(rmse=loo, res=res_loo, acf={k:acf(v) for k,v in res_loo.items()}), open('loo.pkl','wb'))
pd.set_option('display.width',250); print(ff['table'][['model','rate_mm_yr','rate_sd_mm_yr','model_RMSE_cm','acf1','nll','AIC']].to_string())
print(ff['TVS']['params'], ff['StGM_params'])
for k in ['GP','GP2M32','GPM12','GPQP','GP2M32SA']: print(k, {a:round(b,6) for a,b in ff[k]['params'].items()}, ff[k]['beta'])
