from typing import Tuple
import mpmath as mp

def _f(lam, Lx):
    s = mp.sqrt(lam)
    z = s * (Lx/2)
    if abs(s) < mp.mpf('1e-18'):
        return (Lx/2) - (Lx**3 / mp.mpf(48)) * lam
    return mp.tanh(z) / s

def _p_q(
    omega: float, Lx: float,
    cb: float, ct: float, 
    c12: float, g12: float,
    cq1: float, cq2: float,
    sigma1: float, sigma2: float,
    dps: int = 150,
) -> Tuple[float, float]:
    
    mp.dps = dps
    omega = mp.mpf(omega); Lx = mp.mpf(Lx)
    cb = mp.mpf(cb); ct = mp.mpf(ct)
    c12 = mp.mpf(c12); g12 = mp.mpf(g12)
    cq1 = mp.mpf(cq1); cq2 = mp.mpf(cq2)
    sigma1 = mp.mpf(sigma1); sigma2 = mp.mpf(sigma2)
    
    kappa = ct * cb + ct * c12 + cb * c12
    kdenom = kappa + cq2 * (ct + c12) + cq1 * (cb + c12) + cq1*cq2
    kapP = kappa / kdenom
    gamma = 1 + (ct + c12)/cq1 + (cb + c12)/cq2 + kappa/(cq1*cq2)
    xi_rp = ct + cb + kappa * (1/cq1 + 1/cq2)
    xi = 1j * omega * kappa - g12 * xi_rp

    # trace, det, disc, etc. of M
    tr = g12 * (1/sigma1 + 1/sigma2) - (1j * omega* kapP) * (cq1/sigma1 + cq2/sigma2) - \
        (1j * omega * cq1 * cq2 / kdenom) * ((ct + c12)/sigma1 + (cb + c12)/sigma2)
    det = 1j * omega * xi / (gamma * sigma1 * sigma2)
    disc = tr*tr - 4*det
    s = mp.sqrt(disc)

    lam1 = (tr + s) / 2
    lam2 = (tr - s) / 2

    if abs(lam1 - lam2) < mp.mpf('1e-12') * max(abs(lam1), abs(lam2), mp.mpf(1)):        
        lam = tr / 2
        s = mp.sqrt(lam)
        z = s * (Lx / 2)
        f = _f(lam, Lx)
        sech2 = 1/mp.cosh(z)**2
        fprime = Lx / (4 * lam) * sech2 - mp.tanh(z) / (2 * lam**1.5)
        p = fprime
        q = f - p * lam
    else:
        f1 = _f(lam1, Lx)
        f2 = _f(lam2, Lx)
        p = (f1 - f2) / (lam1 - lam2)
        q = f1 - p * lam1
    
    return 2*p/Lx, 2*q/Lx

def calcY(
    omega: float, Lx: float,
    cb: float, ct: float, 
    c12: float, g12: float,
    cq1: float, cq2: float,
    sigma1: float, sigma2: float,
    dps: int = 50,
) -> dict:
    
    mp.dps = dps
    omega = mp.mpf(omega); Lx = mp.mpf(Lx)
    cb = mp.mpf(cb); ct = mp.mpf(ct)
    c12 = mp.mpf(c12); g12 = mp.mpf(g12)
    cq1 = mp.mpf(cq1); cq2 = mp.mpf(cq2)
    sigma1 = mp.mpf(sigma1); sigma2 = mp.mpf(sigma2)

    # Expressions that pop up frequently
    kappa = ct * cb + ct * c12 + cb * c12
    kdenom = kappa + cq2 * (ct + c12) + cq1 * (cb + c12) + cq1*cq2
    kapP = kappa / kdenom
    kapQ = (cq2 * (ct + c12) + cq1 * (cb + c12) + cq1*cq2) / kdenom
    gamma = 1 + (ct + c12)/cq1 + (cb + c12)/cq2 + kappa/(cq1*cq2)
    gtop = 1 + (ct + c12)/cq1 + c12/cq2
    gbot = 1 + (cb + c12)/cq2 + c12/cq1
    xi_rp = ct + cb + kappa * (1/cq1 + 1/cq2)
    xi = 1j * omega * kappa - g12 * xi_rp
    Delta = (kappa / c12) * gtop * gbot - kapQ * xi_rp
    gDeltaFactor = Delta * (g12 / xi)

    diagnostic = dict(
        kappa = kappa, 
        kdenom = kdenom, 
        kapP = kapP, 
        kapQ = kapQ, 
        gamma = gamma,
        gtop = gtop,
        gbot = gbot,
        xi_rp = xi_rp,
        xi = xi,
        Delta = Delta,
        gDeltaFactor = gDeltaFactor,
    )

    p, q = _p_q(omega, Lx, cb, ct, c12, g12, cq1, cq2, sigma1, sigma2, dps = dps)
    diagnostic['p'] = p
    diagnostic['q'] = q

    # Uniform and Direct Contributions
    Y_uniform_direct = kapP

    # Helmholtz Contributions
    Y_particular = kapQ - gDeltaFactor
    Y_helmholtz = (1 - q) * Y_particular + (1j * omega * p * Lx**2 / 4) * \
        (kapP / kdenom) * ((cq1**2/sigma1) * (cb + c12 + cq2) + (cq2**2/sigma2) * (ct + c12 + cq1))

    return dict(
        Y_uniform_direct = Y_uniform_direct,
        Y_particular = Y_particular,
        Y_helmholtz = Y_helmholtz,
        Y_total = Y_uniform_direct + Y_helmholtz,
        diagnostic = diagnostic,
    )
