
"""
pyCoat.py

Author: Ross Johnston

This code is free to use and modify for academic and non-commercial purposes.

"""


import sys
import os
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, leastsq, least_squares, minimize
from scipy.interpolate import interp1d
from scipy import integrate as TG
import scipy.constants as sc
import sqlite3
from pathlib import Path
import ruamel.yaml as ryaml
import tmm

import math as m
import cmath as cm
import scipy.integrate as integrate


def PCI_Dispersion(N,dndT,n,k,rho,c,beta,fc,w0,w1,lam1,IS,source='surf'):
    """

    Determine Signal and Phase from initial parameters of the PCI setup

    LIMITATIONS:
    1. No thermal expansion effects, dn/dT only
    Material properties required for calculations:

    INPUTS
    N = 100 # Default number of integration steps for integrate.Quad 
    & For each material:
    dndT : dn/dT at probe wavelength (10-5)  ,coefficient of thermal expansion should be about or smaller than dn/dT to neglect elasto-optic contributions to dn/dT
    n : index of refraction 
    k : thermal conductivity (W/m-K)
    rho : density (kg/m3)
    c : specific heat (J/kg-K )
    
    For setup:
    beta : Crossing angle in radians (Assuming pump is at normal incidence)
    fc : Modulation frequency
    w0 : Pump waist (um)
    w1 : Probe waist (um),  Assuming, I believe, that the both beams cross at their waist
    lam1 : Probe wavelength (nm)
    IS : Distance between crossing point and photodiode surface (mm)

    method : PCI only for now
    source : surf or bulk

    OUTPUTS
    [0] : Signal
    [1] : phase

    VALIDATION

    Reproduces the same results as the mathematica file from Jessica and the scaling mathcad / word files

    Author: Ross Johnston 2021
    """

    #Normalised parameters

    zt = m.pi*(w0)**2/(2*lam1) # rayleigh range
    ep = 0.5*(w0/w1)**2 #rayleigh ratio
    zeta = IS/zt #norm distance
    tau = (w0**2)*(rho*c)*(1/(8*k))*1e-12

    ft = (1/(2*m.pi*tau)) # thermal relaxation freq
    omg = fc/ft #norm freq

    #Photothermal Coeeficient

    if source=='bulk':
        A = m.sqrt(m.pi/8)*(w0/lam1)*(dndT*n/k)*(1/m.sin(beta)) # Photo-thermal coefficient
        Dispersion = lambda tau: ((1/cm.sqrt(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cm.exp(-complex(0,1)*omg*tau)).real
        Dispersion_im = lambda tau: ((1/cm.sqrt(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cm.exp(-complex(0,1)*omg*tau)).imag
        D,err = integrate.quad(Dispersion,0,N*(m.pi/omg))
        D_im,err_im = integrate.quad(Dispersion_im,0,N*(m.pi/omg))
        D_abs = abs(complex(D,D_im))
        D_arg = np.angle(complex(D,D_im),deg=True)
    else:
        # i.e source == 'surface'
        A = m.sqrt(m.pi/8)*(w0/lam1)*(dndT/k) # # Photothermal-coefficient
        Dispersion = lambda tau: ((1/(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cm.exp(-complex(0,1)*omg*tau)).real
        Dispersion_im = lambda tau: ((1/(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cm.exp(-complex(0,1)*omg*tau)).imag
        D,err = integrate.quad(Dispersion,0,N*(m.pi/omg))
        D_im,err_im = integrate.quad(Dispersion_im,0,N*(m.pi/omg))
        D_abs = abs(complex(D,D_im))
        D_arg = np.angle(complex(D,D_im),deg=True)

    return [A*D_abs,D_arg]
