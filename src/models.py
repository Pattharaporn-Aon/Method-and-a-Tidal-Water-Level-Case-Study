import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, solve_triangular
T0 = pd.Timestamp('2020-01-01')
def tyears(idx): return ((idx - T0).days.values)/365.25

def basis(t, harmonics=(1,)):
    cols=[np.ones_like(t), t]
    for k in harmonics: cols += [np.sin(2*np.pi*k*t), np.cos(2*np.pi*k*t)]
    return np.column_stack(cols)

# ---------------- kernels (r in years) ----------------
def k_m32(r, sf2, ell):
    a = np.sqrt(3)*r/ell; return sf2*(1+a)*np.exp(-a)
def k_exp(r, sf2, ell): return sf2*np.exp(-r/ell)
def k_qp(r, sf2, lp, ld, P=1.0):
    return sf2*np.exp(-2*np.sin(np.pi*r/P)**2/lp**2)*np.exp(-r**2/(2*ld**2))


def _m32(r, lp):
    sf2, ell = np.exp(lp[0]), np.exp(lp[1]); a = np.sqrt(3)*r/ell; e = np.exp(-a)
    K = sf2*(1+a)*e; return K, [K, sf2*a*a*e]
def _exp(r, lp):
    sf2, tau = np.exp(lp[0]), np.exp(lp[1]); K = sf2*np.exp(-r/tau); return K, [K, K*r/tau]
def _qp(r, lp, P=1.0):
    sf2, l1, l2 = np.exp(lp[0]), np.exp(lp[1]), np.exp(lp[2]); s2 = np.sin(np.pi*r/P)**2
    K = sf2*np.exp(-2*s2/l1**2 - r**2/(2*l2**2)); return K, [K, K*4*s2/l1**2, K*r**2/l2**2]
def _sum(*parts):
    def f(r, lp):
        K = 0; G = []; i = 0
        for fn, n in parts:
            k, g = fn(r, lp[i:i+n]); K = K + k; G += g; i += n
        return K, G
    return f
KERNELS = {
 # name: (kernel(r, logparams)->(K, dK/dlogparams), names, initial log-params, basis harmonics)
 'GP':   (_m32, ['sf2','ell_yr'], [np.log(0.01), np.log(0.05)], (1,)),
 'GPQP': (_sum((_m32,2),(_qp,3)), ['sf2_m32','ell_m32_yr','sf2_qp','lp_qp','ld_qp_yr'],
          [np.log(0.008), np.log(0.01), np.log(0.004), np.log(0.8), np.log(3.0)], (1,)),
 'StGM': (_exp, ['sf2','tau_yr'], [np.log(0.01), np.log(0.01)], (1,2)),
}

class GPModel:
    """GP with explicit basis mean h(t)beta (beta profiled by GLS) and kernel hyperparameters by
    maximum likelihood (Xu et al. 2024, GPS Solutions 28:79)."""
    def __init__(self, kind='GP'):
        self.kind = kind; self.kf, self.pnames, self.p0, self.harm = KERNELS[kind]
    def _prep(self, lp, R):
        K, G = self.kf(R, lp[:-1]); sn2 = np.exp(lp[-1])
        K = K.copy(); K[np.diag_indices_from(K)] += sn2 + 1e-8
        return K, G + [sn2*np.eye(len(R))]
    def _nll(self, lp, y, H, R):
        K, G = self._prep(lp, R)
        try: c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError: return 1e10, np.zeros_like(lp)
        KiH = cho_solve(c, H); A = H.T@KiH
        beta = np.linalg.solve(A, KiH.T@y); res = y - H@beta
        alpha = cho_solve(c, res)
        nll = 0.5*res@alpha + np.log(np.diag(c[0])).sum() + 0.5*len(y)*np.log(2*np.pi)
        Ki = cho_solve(c, np.eye(len(y)))
        W = np.outer(alpha, alpha) - Ki
        grad = np.array([-0.5*np.sum(W*g) for g in G])
        return nll, grad
    def fit(self, t, y, starts=None):
        self.t, self.y = t, y; H = basis(t, self.harm); R = np.abs(t[:,None]-t[None,:])
        starts = starts or [np.r_[self.p0, np.log(0.003)]]
        best = None
        for s0 in starts:
            o = minimize(self._nll, np.asarray(s0, float), args=(y,H,R), jac=True, method='L-BFGS-B',
                         bounds=[(-14,4)]*len(s0))
            if best is None or o.fun < best.fun: best = o
        self.opt = best; self.lp = best.x; self.nll = best.fun
        K, _ = self._prep(best.x, R); c = cho_factor(K, lower=True); self.c = c
        KiH = cho_solve(c, H); A = H.T@KiH; self.Ainv = np.linalg.inv(A)
        self.beta = self.Ainv@(KiH.T@y); self.alpha = cho_solve(c, y - H@self.beta)
        self.KiH = KiH; self.H = H
        self.params = dict(zip(self.pnames+['sn2'], np.exp(best.x)))
        self.k = len(best.x) + H.shape[1]
        return self
    def predict(self, ts):
        Hs = basis(ts, self.harm)
        Ks, _ = self.kf(np.abs(ts[:,None]-self.t[None,:]), self.lp[:-1])
        mean = Hs@self.beta + Ks@self.alpha
        kss, _ = self.kf(np.zeros(1), self.lp[:-1])
        V = cho_solve(self.c, Ks.T)                       # n x m
        var = kss[0] - np.einsum('ij,ji->i', Ks, V)
        Rm = Hs - Ks@self.KiH
        var = var + np.einsum('ij,jk,ik->i', Rm, self.Ainv, Rm)
        return mean, np.sqrt(np.maximum(var, 0)), Hs@self.beta
    def rate(self): return self.beta[1], np.sqrt(self.Ainv[1,1])
    def aic(self): return 2*self.nll + 2*self.k

# ---------------- Standard model (OLS, constant annual + semiannual) ----------------
class StModel:
    harm=(1,2)
    def fit(self, t, y):
        H = basis(t, self.harm); self.beta, *_ = np.linalg.lstsq(H, y, rcond=None)
        r = y - H@self.beta; s2 = r@r/(len(y)-H.shape[1]); self.cov = s2*np.linalg.inv(H.T@H)
        self.resid_std = np.sqrt(s2); return self
    def predict(self, ts):
        m = basis(ts, self.harm)@self.beta; return m, np.full(len(ts), self.resid_std), m
    def rate(self): return self.beta[1], np.sqrt(self.cov[1,1])

# ---------------- Bennett-type time-varying seasonal model (Kalman smoother) ----------------
class TVSModel:
    """y_t = a + v t + sum_k [A_k(t) sin + B_k(t) cos] + c_t + e_t
       A_k, B_k: random walks (daily steps); c_t: AR(1) coloured noise; e_t: white noise."""
    harm=(1,2)
    def _Hrow(self, ti):
        return np.r_[basis(np.array([ti]), self.harm)[0], 1.0]
    def _kf(self, lp, t, y, smooth=False):
        q1, q2, r = np.exp(lp[:3]); phi = np.tanh(lp[3]); qc = np.exp(lp[4])
        n = len(t); m = 2+2*len(self.harm)+1
        F = np.eye(m); F[-1,-1] = phi
        Q = np.diag([0,0,q1,q1,q2,q2,qc])
        x = np.zeros(m); P = np.eye(m)*1e2; P[1,1]=1e1; P[-1,-1] = qc/(1-phi**2)
        xf=np.zeros((n,m)); Pf=np.zeros((n,m,m)); xp=np.zeros((n,m)); Pp=np.zeros((n,m,m))
        nll = 0.0; Hs = np.column_stack([basis(t, self.harm), np.ones(n)])
        for i in range(n):
            if i>0: x = F@x; P = F@P@F.T + Q
            xp[i]=x; Pp[i]=P
            if not np.isnan(y[i]):
                Hh = Hs[i]; PH = P@Hh; S = Hh@PH + r; Kg = PH/S; inn = y[i]-Hh@x
                x = x + Kg*inn; P = P - np.outer(Kg, PH)
                if i>60: nll += 0.5*(np.log(2*np.pi*S) + inn**2/S)
            xf[i]=x; Pf[i]=P
        if not smooth: return nll
        xs=xf.copy(); Ps=Pf.copy()
        for i in range(n-2,-1,-1):
            J = Pf[i]@F.T@np.linalg.inv(Pp[i+1])
            xs[i] = xf[i] + J@(xs[i+1]-xp[i+1]); Ps[i] = Pf[i] + J@(Ps[i+1]-Pp[i+1])@J.T
        return xs, Ps, Hs
    def fit(self, t_all, y_all):
        """t_all: regular daily grid (years); y_all with NaN for missing (and masked test values)."""
        self.t, self.y = t_all, y_all
        x0 = np.r_[np.log([1e-7,1e-8,0.001]), np.arctanh(0.7), np.log(0.003)]
        o = minimize(self._kf, x0, args=(t_all,y_all), method='Nelder-Mead',
                     options={'maxiter':600,'xatol':1e-3,'fatol':1e-3})
        self.lp=o.x; self.nll=o.fun
        xs, Ps, Hs = self._kf(o.x, t_all, y_all, smooth=True); self.xs, self.Ps = xs, Ps
        self.mean = np.einsum('ij,ij->i', Hs, xs)                     # signal incl. AR state
        Hd = Hs.copy(); Hd[:,-1] = 0
        self.det = np.einsum('ij,ij->i', Hd, xs)                       # trend + time-varying seasonal
        self.sd = np.sqrt(np.einsum('ij,ijk,ik->i', Hs, Ps, Hs) + np.exp(o.x[2]))
        self.params = {'q_annual':np.exp(o.x[0]),'q_semiannual':np.exp(o.x[1]),'r_white':np.exp(o.x[2]),
                       'phi_AR1':np.tanh(o.x[3]),'q_AR1':np.exp(o.x[4])}
        return self
    def rate(self): return self.xs[-1,1], np.sqrt(self.Ps[-1,1,1])

KERNELS['GPM12'] = (_exp, ['sf2','ell_yr'], [np.log(0.01), np.log(0.01)], (1,))
KERNELS['GP2M32'] = (_sum((_m32,2),(_m32,2)), ['sf2_short','ell_short_yr','sf2_long','ell_long_yr'],
                     [np.log(0.007), np.log(0.005), np.log(0.002), np.log(0.1)], (1,))
KERNELS['GP2M32SA'] = (KERNELS['GP2M32'][0], KERNELS['GP2M32'][1], KERNELS['GP2M32'][2], (1,2))
