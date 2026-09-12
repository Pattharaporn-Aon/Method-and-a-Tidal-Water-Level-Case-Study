import numpy as np, pandas as pd, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':400,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'legend.fontsize':6.2})
COL = {'LI':'#8c8c8c','St':'#1f77b4','TVS':'#9467bd','GP':'#d62728','GP-2M-SA':'#2ca02c'}
d = pd.read_pickle('daily.pkl'); ff = pickle.load(open('final_fit.pkl','rb')); hr = pickle.load(open('hourly_recon.pkl','rb'))
E = pd.read_pickle('exp_results.pkl'); gi = ff['grid_idx']
W = 3.45
# Fig 1
fig, ax = plt.subplots(figsize=(W, 2.0))
w0, w1 = '2021-05-15', '2021-10-15'; sel = (gi>=w0)&(gi<=w1); g = gi[sel]
m, sd = ff['GP2M32SA']['mean'][sel], ff['GP2M32SA']['sd'][sel]
ax.axvspan(pd.Timestamp('2021-06-25'), pd.Timestamp('2021-09-08'), color='#f6e3a1', alpha=.5, lw=0, label='Missing days')
ax.fill_between(g, m-1.96*sd, m+1.96*sd, color=COL['GP-2M-SA'], alpha=.18, lw=0, label='GP-2M-SA 95% interval')
ax.plot(g, ff['St']['mean'][sel], '--', color=COL['St'], lw=.9, label='Standard model')
ax.plot(g, ff['GP']['mean'][sel], color=COL['GP'], lw=.8, label='GP (Matérn 3/2)')
ax.plot(g, m, color=COL['GP-2M-SA'], lw=.9, label='GP-2M-SA mean')
dd = d.loc[w0:w1]; ax.plot(dd.index, dd.dmwl, 'o', ms=1.6, color='k', label='Observed')
ax.set_xlim(pd.Timestamp(w0), pd.Timestamp(w1)); ax.set_ylabel('Daily mean level (m)')
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(ncol=2, frameon=False, loc='upper left', handlelength=1.6, columnspacing=1.0)
ax.set_ylim(2.35, 3.25)
fig.tight_layout(pad=0.3); fig.savefig('/home/claude/gp_paper/fig_gap2021.png'); plt.close()
# Fig 2
order = ['LI','St','TVS','GP','GP-2M-SA']; labs = {'LI':'Linear interp.','St':'Standard','TVS':'Kalman (TVS)','GP':'GP (Matérn 3/2)','GP-2M-SA':'GP-2M-SA'}
groups = [('A_interior_block','A\nblock'),('B_end_prediction','B\nend'),('C_realistic_JunSep_75d','C1\nJun–Sep'),('C_realistic_NovJan_39d','C2\nNov–Jan'),('D_scattered_1to10d','D\nscattered')]
P = E.pivot_table(index='group', columns='model', values='RMSE_cm')
fig, ax = plt.subplots(figsize=(W, 2.05)); x = np.arange(len(groups)); w = .16
for i,k in enumerate(order):
    v = [P.loc[g,k] for g,_ in groups]
    ax.bar(x+(i-2)*w, v, w, color=COL[k], label=labs[k], lw=0)
ax.set_xticks(x); ax.set_xticklabels([l for _,l in groups]); ax.set_ylabel('RMSE of withheld days (cm)')
ax.set_ylim(0, 21); ax.legend(ncol=3, frameon=False, loc='lower left', bbox_to_anchor=(0,1.0), handlelength=1.2, columnspacing=0.8)
ax.axhline(0, color='k', lw=.5)
fig.tight_layout(pad=0.3); fig.savefig('/home/claude/gp_paper/fig_rmse.png'); plt.close()
# Fig 3 hourly
ex = hr['examples'][13]; wv = ex.loc['2022-07-24':'2022-07-31']
fig, ax = plt.subplots(figsize=(W, 1.55))
ax.plot(wv.index, wv.obs, color='k', lw=.9, label='Observed (withheld)')
ax.plot(wv.index, wv['Tide + GP-2M-SA'], color=COL['GP-2M-SA'], lw=.9, label='Tide + GP-2M-SA')
ax.plot(wv.index, wv['LI (hourly)'], '--', color=COL['LI'], lw=.9, label='Linear interpolation')
ax.set_ylabel('Water level (m)'); ax.set_ylim(0.8, 4.9)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(ncol=3, frameon=False, loc='lower left', bbox_to_anchor=(0,0.98), handlelength=1.4, columnspacing=0.7, fontsize=5.8)
fig.tight_layout(pad=0.3); fig.savefig('/home/claude/gp_paper/fig_hourly.png'); plt.close()
print('ok')
