import numpy as np, pandas as pd, pickle, sys, time
from experiments import *
C = cases()
for ci in [int(a) for a in sys.argv[1:]]:
    tic=time.time(); c=C[ci]; test = c['mask'] & obs; train = obs & ~c['mask']
    g = GPModel('GP2M32SA').fit(t_all[train], y_all[train]); m, sd, _ = g.predict(t_all[test])
    sdy = np.sqrt(sd**2 + g.params['sn2'])
    row = dict(model='GP-2M-SA', **metrics(y_all[test], m, sdy), rate_mm_yr=1000*g.rate()[0], case=ci, group=c['group'], label=c['label'], pct=c['pct'], n_test=int(test.sum()), n_train=int(train.sum()))
    pickle.dump((row, m, sdy), open(f'exp/sa_{ci:02d}.pkl','wb')); print(ci, round(row['RMSE_cm'],2), f'{time.time()-tic:.0f}s', flush=True)
