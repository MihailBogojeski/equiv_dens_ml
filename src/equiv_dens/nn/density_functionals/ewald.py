import torch
import torch.nn as nn


def ewald_energy(rho=None, pos, a_num, PME=False):
    prec = 1.0e-8


def get_best_eta(precision, gmax, pos, a_num):
    # charge
    charge = 0.0
    chargeSquare = 0.0
    for i in np.arange(len(ions.pos)):
        charge += ions.Zval[ions.labels[i]]
        chargeSquare += ions.Zval[ions.labels[i]] ** 2

    # eta
    eta = 1.6
    NotGoodEta = True
    while NotGoodEta:
        # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
        upbound = (
            4.0 * np.pi * ions.nat * chargeSquare * np.sqrt(eta / np.pi) * sp.erfc(gmax / 2.0 * np.sqrt(1.0 / eta))
        )
        if upbound < precision:
            NotGoodEta = False
        else:
            eta = eta - 0.01
    return eta
