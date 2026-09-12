import numpy as np, pandas as pd, pickle, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
COL = {'obs':'#222222','St':'#1f77b4','TVS':'#9467bd','GP':'#d62728','GP-2M':'#ff7f0e','GP-2M-SA':'#2ca02c','LI':'#7f7f7f'}
d = pd.read_pickle('daily.pkl'); ff = pickle.load(open('final_fit.pkl','rb')); hr = pickle.load(open('hourly_recon.pkl','rb'))
s10 = pd.read_pickle('clean_10min.pkl'); raw = pd.read_pickle('../bpk.pkl'); raw.columns=['t','h']
gidx = ff['grid_idx']; n = len(d); obs = d.dmwl.notna().values
gaps = pd.read_csv('daily_gaps.csv', parse_dates=['start','end'])
def shade(ax, minlen=3):
    for _,g in gaps[gaps.days>=minlen].iterrows(): ax.axvspan(g.start, g.end+pd.Timedelta('1D'), color='#f4d03f', alpha=.35, lw=0)
# Fig 1 overview
fig, ax = plt.subplots(2,1, figsize=(7.2,5.0), sharex=True, gridspec_kw={'height_ratios':[1.3,1]})
removed = raw.drop_duplicates('t'); removed = removed[~removed.set_index('t').index.isin(s10.index)]
s10b = s10.copy(); br = s10b.index.to_series().diff() > pd.Timedelta('1h'); s10b[br[br].index] = np.nan
ax[0].plot(s10b.index, s10b.values, lw=.15, color='#5dade2', label='10-min water level (after QC)')
ax[0].scatter(removed.t, removed.h, s=2, color='#c0392b', label=f'Removed by QC (n = {len(removed):,})', zorder=3)
shade(ax[0]); ax[0].set_ylabel('Water level (m)'); ax[0].legend(loc='lower left', bbox_to_anchor=(0,1.0), fontsize=7, frameon=False, ncol=2, markerscale=3)
ax[1].plot(d.index, d.dmwl, '.', ms=1.5, color=COL['obs'], label='Daily mean water level (tide-filtered)')
shade(ax[1]); ax[1].set_ylabel('Daily mean (m)'); ax[1].legend(loc='upper left', fontsize=7, frameon=False)
ax[1].xaxis.set_major_locator(mdates.YearLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
for a,l in zip(ax,'ab'): a.text(-0.08,1.0,f'({l})',transform=a.transAxes,fontweight='bold')
fig.tight_layout(); fig.savefig('fig/Fig1_data_overview.png'); plt.close()
# Fig 2 model fits over full record
fig, ax = plt.subplots(3,1, figsize=(7.2,6.6), sharex=True, sharey=True)
for a, key, lab in zip(ax, ['St','TVS','GP'], ['Standard model','Bennett-type model (Kalman)','GP model (Matérn 3/2)']):
    a.plot(d.index, d.dmwl, '.', ms=1.3, color='#888888', label='Observed')
    if key=='St': a.plot(gidx, ff['St']['mean'], color=COL['St'], lw=1, label=lab)
    elif key=='TVS': a.plot(d.index, ff['TVS']['mean'], color=COL['TVS'], lw=.7, label=lab)
    else:
        m, sd = ff['GP']['mean'], ff['GP']['sd']
        a.fill_between(gidx, m-1.96*sd, m+1.96*sd, color=COL['GP'], alpha=.18, lw=0, label='95% interval')
        a.plot(gidx, m, color=COL['GP'], lw=.7, label=lab)
        a.axvline(d.index[-1], color='k', ls=':', lw=.8)
    shade(a); a.legend(loc='upper left', fontsize=7, frameon=False, ncol=3); a.set_ylabel('Daily mean (m)')
ax[-1].xaxis.set_major_locator(mdates.YearLocator()); ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
for a,l in zip(ax,'abc'): a.text(-0.08,1.0,f'({l})',transform=a.transAxes,fontweight='bold')
fig.tight_layout(); fig.savefig('fig/Fig2_model_fits.png'); plt.close()
# Fig 3 leave-one-out residuals + ACF
LO = pickle.load(open('loo.pkl','rb'))
fig = plt.figure(figsize=(7.2,5.2)); gs = fig.add_gridspec(3,2, width_ratios=[3,1.3], wspace=0.35)
di = d.index[obs]
labs = {'St':'Standard','TVS':'Bennett-type','GP':'GP (Matérn 3/2)','GP-2M':'GP-2M','GP-2M-SA':'GP-2M-SA','LI':'Linear interpolation'}
key = {'St':'St','GP':'GP','GP-2M-SA':'GP2M32SA'}
for i,k in enumerate(key):
    r = LO['res'][key[k]]
    a = fig.add_subplot(gs[i,0]); rr = pd.Series(100*r, di).reindex(d.index); a.plot(d.index, rr.values, lw=.4, color=COL[k]); a.set_ylim(-45,45); a.set_ylabel('cm')
    a.text(0.01,0.86,f'{labs[k]}: leave-one-out RMSE = {100*np.sqrt(np.mean(r**2)):.1f} cm', transform=a.transAxes, fontsize=7.5)
    a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if i<2: a.set_xticklabels([])
ab = fig.add_subplot(gs[:,1])
for k in key: ab.plot(np.arange(1,61), LO['acf'][key[k]], color=COL[k], lw=1, label=labs[k])
ab.axhline(0,color='k',lw=.5); ab.axhspan(-1.96/np.sqrt(obs.sum()),1.96/np.sqrt(obs.sum()),color='grey',alpha=.2)
ab.set_xlabel('Lag (days)'); ab.set_ylabel('Autocorrelation of LOO residuals'); ab.legend(fontsize=7, frameon=False)
fig.subplots_adjust(left=0.08,right=0.98,top=0.97,bottom=0.09,wspace=0.32,hspace=0.22); fig.savefig('fig/Fig3_residuals_acf.png'); plt.close()
# Fig 4 reconstruction of real gaps
wins = [('2021-05-20','2021-10-10','Jun–Sep 2021 gap (75 days)'), ('2020-10-25','2021-02-05','Nov 2020–Jan 2021 gap (39 days)'), ('2019-10-05','2019-12-25','Nov 2019 gap (25 days)')]
fig, ax = plt.subplots(3,1, figsize=(7.2,7.0))
for a,(w0,w1,title) in zip(ax,wins):
    sel = (gidx>=w0)&(gidx<=w1); gs_ = gidx[sel]
    for k,kk in (('GP','GP'),('GP-2M-SA','GP2M32SA')):
        m, sd = ff[kk]['mean'][sel], ff[kk]['sd'][sel]
        a.fill_between(gs_, m-1.96*sd, m+1.96*sd, color=COL[k], alpha=.13, lw=0)
        a.plot(gs_, m, color=COL[k], lw=1, label=f'{labs[k]} posterior mean ± 95%')
    a.plot(gs_, ff['St']['mean'][sel], color=COL['St'], lw=1, ls='--', label='Standard model')
    dd = d.loc[w0:w1]; a.plot(dd.index, dd.dmwl, 'o', ms=2, color='k', label='Observed')
    shade(a, 2); a.set_xlim(pd.Timestamp(w0), pd.Timestamp(w1)); a.set_title(title, fontsize=9, loc='left'); a.set_ylabel('Daily mean (m)')
    a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax[0].legend(fontsize=7, frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0,1.08))
fig.tight_layout(); fig.savefig('fig/Fig4_gap_reconstruction.png'); plt.close()
# Fig 5 experiments
E = pd.read_pickle('exp_results.pkl')
fig, ax = plt.subplots(2,2, figsize=(7.2,5.6))
order = ['LI','St','TVS','GP','GP-2M-SA']
for a, grp, title in zip(ax.flat[:3], ['A_interior_block','B_end_prediction','D_scattered_1to10d'],
                         ['(a) Contiguous interior gap','(b) End-of-record prediction','(c) Scattered 1–10-day gaps']):
    sub = E[E.group==grp].groupby(['pct','model']).RMSE_cm.mean().unstack()
    for k in order:
        if k in sub: a.plot(sub.index, sub[k], 'o-', color=COL[k], lw=1.2, ms=4, label=labs.get(k, 'Linear interp.' if k=='LI' else k) if k!='LI' else ('Persistence' if grp=='B_end_prediction' else 'Linear interpolation'))
    a.set_title(title, fontsize=9, loc='left'); a.set_xlabel('Missing data (%)'); a.set_ylabel('RMSE (cm)'); a.set_xticks([10,20,30,40])
ax[0,0].legend(fontsize=7, frameon=False)
a = ax[1,1]; sub = E[E.group.str.startswith('C_')].copy()
sub['win'] = np.where(sub.group.str.contains('JunSep'),'Jun–Sep (75 d)','Nov–Jan (39 d)')
piv = sub.groupby(['win','model']).RMSE_cm.mean().unstack()[order]
x = np.arange(len(piv)); w = .16
for i,k in enumerate(order): a.bar(x+(i-2)*w, piv[k], w, color=COL[k], label={'LI':'LI','St':'Standard','TVS':'Bennett-type','GP':'GP','GP-2M-SA':'GP-2M-SA'}[k])
a.set_xticks(x); a.set_xticklabels(piv.index); a.set_ylabel('Mean RMSE (cm)'); a.set_title('(d) Realistic gap windows (5 years each)', fontsize=9, loc='left')
a.set_ylim(0,16); a.legend(fontsize=6.5, frameon=False, ncol=3, loc='upper left')
fig.tight_layout(); fig.savefig('fig/Fig5_gap_recovery_experiments.png'); plt.close()
# Fig 6 hourly validation + reconstruction
ex = hr['examples'][13]; H = hr['hourly']
fig, ax = plt.subplots(3,1, figsize=(7.2,6.8))
w = ex.loc['2022-07-20':'2022-08-03']
ax[0].plot(w.index, w.obs, color='k', lw=1, label='Observed (withheld)')
ax[0].plot(w.index, w['Tide + GP-2M-SA'], color=COL['GP-2M-SA'], lw=1, label='Tide + GP-2M-SA')
ax[0].plot(w.index, w['LI (hourly)'], color=COL['LI'], lw=1, ls='--', label='Linear interpolation')
ax[0].set_ylabel('Water level (m)'); ax[0].set_ylim(0.8,4.9); ax[0].legend(fontsize=7, frameon=False, ncol=3, loc='upper left'); ax[0].set_title('(a) Validation: hourly values withheld 25 Jun–7 Sep 2022 (two-week excerpt)', fontsize=9, loc='left')
ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
e = ex['Tide + GP-2M-SA']-ex.obs
ax[1].hist(100*e, bins=60, color=COL['GP-2M-SA'], alpha=.8); ax[1].set_xlabel('Hourly error (cm)'); ax[1].set_ylabel('Count')
ax[1].set_title(f'(b) Hourly error, Tide + GP-2M-SA: RMSE = {100*np.sqrt((e**2).mean()):.1f} cm, bias = {100*e.mean():.1f} cm', fontsize=9, loc='left')
w = H.loc['2021-06-18':'2021-07-08']
ax[2].plot(w.index, w.observed_m, color='k', lw=1, label='Observed')
m = w.flag>0
ax[2].plot(w.index, w.reconstructed_m.where(w.flag>0), color=COL['GP-2M-SA'], lw=1, label='Reconstructed')
ax[2].fill_between(w.index, (w.reconstructed_m-1.96*w.sd_m).where(m), (w.reconstructed_m+1.96*w.sd_m).where(m), color=COL['GP-2M-SA'], alpha=.15, lw=0)
ax[2].set_ylabel('Water level (m)'); ax[2].set_ylim(0.2,4.9); ax[2].legend(fontsize=7, frameon=False, ncol=2, loc='upper left'); ax[2].set_title('(c) Reconstruction at the start of the Jun–Sep 2021 gap', fontsize=9, loc='left')
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=3)); ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.tight_layout(); fig.savefig('fig/Fig6_hourly_reconstruction.png'); plt.close()
print('figures done')
