#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.integrate import cumtrapz
from scipy import sqrt

from obspy.geodetics import gps2dist_azimuth

from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()


# !----------------------!
# !  set some functions  !
# !----------------------!

def quadgk(func, a, b):
    """ Gauss-Kronrod quadrature (G7-K15)
        func: the integrand
        a, b: lower and upper limits of the integral
    """

    # set Kronrod nodes on [-1, 1] interval
    x_kr = [-9.9145537112081263920685469752598e-01,
            -9.4910791234275852452618968404809e-01,
            -8.6486442335976907278971278864098e-01,
            -7.4153118559939443986386477328110e-01,
            -5.8608723546769113029414483825842e-01,
            -4.0584515137739716690660641207707e-01,
            -2.0778495500789846760068940377309e-01,
            0.0,
            2.0778495500789846760068940377309e-01,
            4.0584515137739716690660641207707e-01,
            5.8608723546769113029414483825842e-01,
            7.4153118559939443986386477328110e-01,
            8.6486442335976907278971278864098e-01,
            9.4910791234275852452618968404809e-01,
            9.9145537112081263920685469752598e-01]

    # set weights
    w_kr = [2.2935322010529224963732008059913e-02,
            6.3092092629978553290700663189093e-02,
            1.0479001032225018383987632254189e-01,
            1.4065325971552591874518959051021e-01,
            1.6900472663926790282658342659795e-01,
            1.9035057806478540991325640242055e-01,
            2.0443294007529889241416199923466e-01,
            2.0948214108472782801299917489173e-01,
            2.0443294007529889241416199923466e-01,
            1.9035057806478540991325640242055e-01,
            1.6900472663926790282658342659795e-01,
            1.4065325971552591874518959051021e-01,
            1.0479001032225018383987632254189e-01,
            6.3092092629978553290700663189093e-02,
            2.2935322010529224963732008059913e-02]

    c1, c2 = (b-a)/2.0, (b+a)/2.0
    eval_points = map(lambda x: c1*x + c2, x_kr)

    func_evals = map(func, eval_points)
    func_evals = np.array(map(list, zip(*func_evals)))

    return c1 * np.sum(func_evals*w_kr, axis=-1)


def quadl(func, a, b):
    """ Gauss-Legendre quadrature, Lobatto rules with n=15
        func: the integrand
        a, b: lower and upper limits of the integral
    """

    # set nodes
    x_lo = [-1.0,
            -0.965245926503838572795851392070,
            -0.885082044222976298825401631482,
            -0.763519689951815200704118475976,
            -0.606253205469845711123529938637,
            -0.420638054713672480921896938739,
            -0.215353955363794238225679446273,
            0.0,
            0.215353955363794238225679446273,
            0.420638054713672480921896938739,
            0.606253205469845711123529938637,
            0.763519689951815200704118475976,
            0.885082044222976298825401631482,
            0.965245926503838572795851392070,
            1.0]

    # set weights
    w_lo = [9.52380952380952380952380952381e-03,
            5.80298930286012490968805840253e-02,
            1.01660070325718067603666170789e-01,
            1.40511699802428109460446805644e-01,
            1.72789647253600949052077099408e-01,
            1.96987235964613356092500346507e-01,
            2.11973585926820920127430076977e-01,
            2.17048116348815649514950214251e-01,
            2.11973585926820920127430076977e-01,
            1.96987235964613356092500346507e-01,
            1.72789647253600949052077099408e-01,
            1.40511699802428109460446805644e-01,
            1.01660070325718067603666170789e-01,
            5.80298930286012490968805840253e-02,
            9.52380952380952380952380952381e-03]

    c1, c2 = (b-a)/2.0, (b+a)/2.0
    eval_points = map(lambda x: c1*x + c2, x_lo)

    func_evals = map(func, eval_points)
    func_evals = np.array(map(list, zip(*func_evals)))

    return c1 * np.sum(func_evals*w_lo, axis=-1)


def harms(evt_lat, evt_lon, z_evt, sta_lat, sta_lon, z_sta,
          Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, time, dt):
    """ compute prompt gravity perturbations induced by
    a double-couple buried in a homogeneous half-space,
    based on analytical solutions derived in Harms (2016) """

    # x = East, y = North, z = Upward

    # compute epicentral distance and azimuth
    epi, faz, _ = gps2dist_azimuth(evt_lat, evt_lon, sta_lat, sta_lon,
                                   a=rt, f=0.0)

    # compute polar angle of horizontal coordinates
    phi = (90.0 - faz) * deg2rad

    # compute distance from event to station
    dis = sqrt(epi**2 + z_evt**2)

    # set moment tensor (East, North, Up)
    M = np.array([[Mpp, -Mtp, Mrp], [-Mtp, Mtt, -Mrt], [Mrp, -Mrt, Mrr]])


    # !--------------------------!
    # !  set internal functions  !
    # !--------------------------!

    # gravity perturbation in transform domain (DoubleCouple_HalfSpacenb)
    # include possibility for sign flip for branch-cut navigation
    eta_i = lambda p, q, si: si*sqrt(q**2 - p**2)
    eta_a = lambda p, q, sa: sa*sqrt(1.0/alpha**2 + q**2 - p**2)
    eta_b = lambda p, q: sqrt(1.0/beta**2 + q**2 - p**2)

    Rp = lambda p, q, sa: (eta_b(p, q)**2 + (q**2-p**2))**2 -\
            4.0*(q**2-p**2)*eta_a(p, q, sa)*eta_b(p, q)

    def xI(M, p, q, phi, si):
        a0 = - (M[0,0]*p**3+M[1,1]*p*q**2)*np.cos(phi)**3

        b1 = M[0,2]*eta_i(p,q,si)
        b2 = M[0,1]*p*np.sin(phi)
        b0 = 2.0*q**2*np.sin(phi)**2*(b1 + b2)

        c1 = M[0,2]*p*eta_i(p,q,si)
        c2 = M[0,1]*(p**2+2*q**2)*np.sin(phi)
        c0 = - 2.0*p*np.cos(phi)**2*(c1 + c2)

        d1 = M[2,2]*p*(p**2-q**2)
        d2 = - 2.0*M[1,2]*eta_i(p,q,si)*(p**2+q**2)*np.sin(phi)
        d3 = - p*(M[1,1]*p**2-3.0*M[0,0]*q**2+2.0*M[1,1]*q**2)*np.sin(phi)**2
        d0 = np.cos(phi)*(d1 + d2 + d3)

        out = -4.0j*np.pi*G*(a0 + b0 + c0 + d0)/eta_i(p,q,si)
        return out

    def yI(M, p, q, phi, si):
        a0 = 2.0*M[0,1]*p*q**2*np.cos(phi)**3

        b1 = 2.0*M[1,2]*q**2*eta_i(p,q,si)
        b2 = - p*(-3.0*M[1,1]*q**2+M[0,0]*(p**2+2.0*q**2))*np.sin(phi)
        b0 = np.cos(phi)**2*(b1 + b2)

        c1 = M[2,2]*(p**2-q**2)
        c2 = - 2.0*M[1,2]*p*eta_i(p,q,si)*np.sin(phi)
        c3 = (-M[1,1]*p**2+M[0,0]*q**2)*np.sin(phi)**2
        c0 = p*np.sin(phi)*(c1 + c2 + c3)

        d1 = M[0,2]*eta_i(p,q,si)*(p**2+q**2)
        d2 = M[0,1]*p*(p**2+2.0*q**2)*np.sin(phi)
        d0 = -np.sin(2.0*phi)*(d1 + d2)

        out = -4.0j*np.pi*G*(a0 + b0 + c0 + d0)/eta_i(p,q,si)
        return out

    def zI(M, p, q, phi, si):
        a0 = M[2,2]*(p**2-q**2)
        b0 = (-M[0,0]*p**2+M[1,1]*q**2)*np.cos(phi)**2
        c0 = - 2.0*M[1,2]*p*eta_i(p,q,si)*np.sin(phi)
        d0 = (-M[1,1]*p**2+M[0,0]*q**2)*np.sin(phi)**2

        e1 = M[0,2]*p*eta_i(p,q,si)
        e2 = M[0,1]*(p**2+q**2)*np.sin(phi)
        e0 = - 2.0*np.cos(phi)*(e1 + e2)

        out = -4.0j*np.pi*G*(a0 + b0 + c0 + d0 + e0)
        return out

    nI = lambda p, q, phi, si: np.array([xI(M,p,q,phi,si),
                                         yI(M,p,q,phi,si),
                                         zI(M,p,q,phi,si)])

    fP = lambda p, q, si, sa: -8.0j*np.pi*G*(eta_i(p,q,si)-eta_b(p,q))**2/Rp(p,q,sa)

    def xP(M, p, q, phi, si, sa):
        a0 = p*(M[0,0]*p**2-M[1,1]*q**2)*np.cos(phi)**3
        b0 = - 2.0*M[0,2]*q**2*eta_a(p,q,sa)*np.sin(phi)**2
        c0 = - 2.0*M[0,1]*p*q**2*np.sin(phi)**3

        d1 = M[0,2]*p*eta_a(p,q,sa)
        d2 = M[0,1]*(p**2+2.0*q**2)*np.sin(phi)
        d0 = 2.0*p*np.cos(phi)**2*(d1 + d2)

        e1 = M[2,2]*eta_a(p,q,sa)**2
        e2 = (M[1,1]*p**2-3.0*M[0,0]*q**2)*np.sin(phi)**2
        e0 = p*np.cos(phi)*(e1 + e2)

        f0 = M[1,2]*(p**2+q**2)*eta_a(p,q,sa)*np.sin(2.0*phi)
        g0 = M[1,1]*p*q**2*np.sin(phi)*np.sin(2.0*phi)

        out = fP(p,q,si,sa)*eta_i(p,q,si)*(a0 + b0 + c0 + d0 + e0 + f0 + g0)
        return out

    def yP(M, p, q, phi, si, sa):
        a0 = -2.0*M[0,1]*p*q**2*np.cos(phi)**3
        b0 = 2.0*M[1,2]*p**2*eta_a(p,q,sa)*np.sin(phi)**2
        c0 = 4.0*M[0,1]*p*q**2*np.cos(phi)*np.sin(phi)**2
        d0 = p*(M[1,1]*p**2-M[0,0]*q**2)*np.sin(phi)**3

        e1 = -2.0*M[1,2]*q**2*eta_a(p,q,sa)
        e2 = p*(M[0,0]*p**2+2.0*M[0,0]*q**2-3.0*M[1,1]*q**2)*np.sin(phi)
        e0 = np.cos(phi)**2*(e1 + e2)

        f0 = M[0,2]*(p**2+q**2)*eta_a(p,q,sa)*np.sin(2.0*phi)

        g1 = M[2,2]*eta_a(p,q,sa)**2
        g2 = M[0,1]*p**2*np.sin(2.0*phi)
        g0 = p*np.sin(phi)*(g1 + g2)

        out = fP(p,q,si,sa)*eta_i(p,q,si)*(a0 + b0 + c0 + d0 + e0 + f0 + g0)
        return out

    def zP(M, p, q, phi, si, sa):
        a0 = M[2,2]*eta_a(p,q,sa)**2
        b0 = 2.0*M[0,2]*p*eta_a(p,q,sa)*np.cos(phi)
        c0 = (M[0,0]*p**2-M[1,1]*q**2)*np.cos(phi)**2
        d0 = 2.0*M[1,2]*p*eta_a(p,q,sa)*np.sin(phi)
        e0 = (M[1,1]*p**2-M[0,0]*q**2)*np.sin(phi)**2
        f0 = M[0,1]*p**2*np.sin(2.0*phi)
        g0 = M[0,1]*q**2*np.sin(2.0*phi)

        out = fP(p,q,si,sa)*(q**2-p**2)*(a0 + b0 + c0 + d0 + e0 + f0 + g0)
        return out

    nP = lambda p, q, phi, si, sa: np.array([xP(M,p,q,phi,si,sa),
                                             yP(M,p,q,phi,si,sa),
                                             zP(M,p,q,phi,si,sa)])

    fS = lambda p, q, si,sa: -8.0j*np.pi*G*(2.0*eta_a(p,q,sa)*eta_i(p,q,si)-(q**2-p**2)-eta_b(p,q)**2)/Rp(p,q,sa)

    def xS(M, p, q, phi, si, sa):
        a0 = (-M[0,0]*p**3*eta_b(p,q)+M[1,1]*p*q**2*eta_b(p,q))*np.cos(phi)**3

        b1 = M[0,2]*(-p**2+q**2+eta_b(p,q)**2)
        b2 = 2.0*M[0,1]*p*eta_b(p,q)*np.sin(phi)
        b0 = q**2*np.sin(phi)**2*(b1 + b2)

        c1 = M[0,2]*p*(-p**2+q**2+eta_b(p,q)**2)
        c2 = 2.0*M[0,1]*(p**2+2.0*q**2)*eta_b(p,q)*np.sin(phi)
        c0 = - p*np.cos(phi)**2*(c1 + c2)

        d1 = M[2,2]*p*(p**2-q**2)*eta_b(p,q)
        d2 = M[1,2]*(p**2+q**2)*(p**2-q**2-eta_b(p,q)**2)*np.sin(phi)
        d3 = -p*(M[1,1]*p**2-3.0*M[0,0]*q**2+2.0*M[1,1]*q**2)*eta_b(p,q)*np.sin(phi)**2
        d0 = np.cos(phi)*(d1 + d2 + d3)

        out = fS(p,q,si,sa)*(a0 + b0 + c0 + d0)
        return out

    def yS(M, p, q, phi, si, sa):
        a0 = 2.0*M[0,1]*p*q**2*eta_b(p,q)*np.cos(phi)**3

        b1 = M[0,2]*(-p**4+q**4+p**2*eta_b(p,q)**2+q**2*eta_b(p,q)**2)
        b2 = 2.0*M[0,1]*p*(p**2+2.0*q**2)*eta_b(p,q)*np.sin(phi)
        b0 = - np.cos(phi)*np.sin(phi)*(b1 + b2)

        c1 = M[1,2]*q**2*(-p**2+q**2+eta_b(p,q)**2)
        c2 = -p*(M[0,0]*p**2+2.0*M[0,0]*q**2-3.0*M[1,1]*q**2)*eta_b(p,q)*np.sin(phi)
        c0 = np.cos(phi)**2*(c1 + c2)

        d1 = M[2,2]*(p**2-q**2)*eta_b(p,q)
        d2 = M[1,2]*p*(p**2-q**2-eta_b(p,q)**2)*np.sin(phi)
        d3 = (-M[1,1]*p**2*eta_b(p,q)+M[0,0]*q**2*eta_b(p,q))*np.sin(phi)**2
        d0 = p*np.sin(phi)*(d1 + d2 + d3)

        out = fS(p,q,si,sa)*(a0 + b0 + c0 + d0)
        return out

    def zS(M, p, q, phi, si, sa):
        a0 = -M[2,2]*(q**2-p**2)*eta_b(p,q)

        b1 = -M[0,0]*p**2*eta_b(p,q)
        b2 = M[1,1]*q**2*eta_b(p,q)
        b0 = np.cos(phi)**2*(b1 + b2)

        c0 = -M[1,2]*p*((q**2-p**2)+eta_b(p,q)**2)*np.sin(phi)
        d0 = (-M[1,1]*p**2*eta_b(p,q)+M[0,0]*q**2*eta_b(p,q))*np.sin(phi)**2

        e1 = M[0,2]*p*((q**2-p**2)+eta_b(p,q)**2)
        e2 = 2.0*M[0,1]*(p**2+q**2)*eta_b(p,q)*np.sin(phi)
        e0 = -np.cos(phi)*(e1 + e2)

        out = fS(p,q,si,sa)*eta_i(p,q,si)*(a0 + b0 + c0 + d0 + e0)
        return out

    nS = lambda p, q, phi, si, sa: np.array([xS(M,p,q,phi,si,sa),
                                             yS(M,p,q,phi,si,sa),
                                             zS(M,p,q,phi,si,sa)])


    # !----------------------!
    # !  Cagniard - de Hoop  !
    # !----------------------!

    # alpha, beta, inf. branch points
    pb1 = lambda q: q
    pb2 = lambda q: sqrt(1.0/alpha**2 + q**2)
    pb3 = lambda q: sqrt(1.0/beta**2 + q**2)

    # Cagniard paths
    p1 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*sqrt(t**2-(pb1(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)
    p2 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*sqrt(t**2-(pb2(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)
    p3 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*sqrt(t**2-(pb3(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)

    dpdt1 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/sqrt(t**2-(pb1(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)
    dpdt2 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/sqrt(t**2-(pb2(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)
    dpdt3 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/sqrt(t**2-(pb3(q)*sqrt(r**2+z**2))**2))/(r**2+z**2)


    # !--------------------------------------------!
    # !  routines : contributions from hyperbolae  !
    # !--------------------------------------------!

    def hI(q, r, phi, z, t):
        bool = 1.0
        diff = nI(p1(r,z,t,q),q,phi,1)*dpdt1(r,z,t,q) -\
                nI(np.conj(p1(r,z,t,q)),q,phi,1)*np.conj(dpdt1(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2

    def hP(q, r, phi, z, t):
        bool = (t > sqrt(r**2+z**2)/alpha)
        diff = nP(p2(r,z,t,q),q,phi,1,1)*dpdt2(r,z,t,q) -\
                nP(np.conj(p2(r,z,t,q)),q,phi,1,1)*np.conj(dpdt2(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2

    def hS(q, r, phi, z, t):
        bool = (t > sqrt(r**2+z**2)/beta)
        diff = nS(p3(r,z,t,q),q,phi,1,1)*dpdt3(r,z,t,q) -\
                nS(np.conj(p3(r,z,t,q)),q,phi,1,1)*np.conj(dpdt3(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2


    # !-------------------------------------------------------!
    # !  routines : contributions from branch-cut navigation  !
    # !-------------------------------------------------------!

    ba = sqrt(1.0/beta**2-1.0/alpha**2)

    def brPa(q, r, phi, z, t):
        bool = (t > np.abs(z)/alpha)*(t < sqrt(r**2+z**2)/alpha)
        diff = nP(p2(r,z,t,q), q, phi, 1, 1) - nP(p2(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt2(r,z,t,q)) / (2.0*np.pi)**2

    def brPb(q, r, phi, z, t):
        bool = (t > sqrt(r**2+z**2)/alpha)*(t < (r**2+z**2)/(alpha*np.abs(z)))
        diff = nP(p2(r,z,t,q), q, phi, 1, 1) - nP(p2(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt2(r,z,t,q)) / (2.0*np.pi)**2

    def brSa(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t > np.abs(z)/beta)
        bool = bool * (t < r/alpha+ba*np.abs(z))
        diff = (nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1))
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSb(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t > r/alpha+ba*np.abs(z))
        bool = bool * (t < np.abs(z)/beta +\
                       r/np.abs(z)*sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSc(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t < (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t > np.abs(z)/beta+\
                       r/np.abs(z)*sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSd(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t > r/alpha+ba*np.abs(z))
        bool = bool * (t < sqrt(r**2+z**2)/beta)
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, -1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSe(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t > sqrt(r**2+z**2)/beta)
        bool = bool * (t < (r**2+z**2)/np.abs(z)*ba)
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, -1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSf(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t < (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t > np.abs(z)/beta+\
                       r/np.abs(z)*sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSg(q, r, phi, z, t):
        bool = (r/sqrt(r**2+z**2) > beta/alpha)*(t > (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t < (r**2+z**2)/(beta*np.abs(z)))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2


    # !----------------------------!
    # !  sum of all contributions  !
    # !----------------------------!

    off = 1.0e-6

    hvI = np.zeros((time.size, 3))
    hvP = np.zeros((time.size, 3))
    hvS = np.zeros((time.size, 3))

    brvP = np.zeros((time.size, 3))
    brvS = np.zeros((time.size, 3))

    for k, tau in enumerate(time):

        # contributions from hyperbolae
        hIt = lambda q: hI(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = tau/dis - off
        hvI[k,:] = np.real(quadgk(hIt, slowness1, slowness2))

        hPt = lambda q: hP(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = sqrt((tau/dis)**2 - 1.0/alpha**2) - off
        hvP[k,:] = np.real(quadgk(hPt, slowness1, slowness2))

        hSt = lambda q: hS(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = sqrt((tau/dis)**2 - 1.0/beta**2) - off
        hvS[k,:] = np.real(quadgk(hSt, slowness1, slowness2))

        # contributions from branch-cut navigation
        brPat = lambda q: brPa(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = (tau - np.abs(z_evt)/alpha) / epi
        brvPa = quadgk(brPat, slowness1, slowness2)

        brPbt = lambda q: brPb(q, epi, phi, z_evt, tau)
        slowness1 = sqrt((tau/dis)**2 - 1.0/alpha**2)
        slowness2 = (tau - np.abs(z_evt)/alpha) / epi
        brvPb = quadgk(brPbt, slowness1, slowness2)

        brvP[k,:] = np.real(brvPa + brvPb)

        brSat = lambda q: brSa(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = (tau - np.abs(z_evt)/beta) / epi
        brvSa = quadl(brSat, slowness1, slowness2)

        brSbt = lambda q: brSb(q, epi, phi, z_evt, tau)
        slowness1 = sqrt((tau-np.abs(z_evt)*ba)**2 - epi**2/alpha**2) / epi
        slowness2 = (tau - np.abs(z_evt)/beta) / epi
        brvSb = quadl(brSbt, slowness1, slowness2)

        brSct = lambda q: brSc(q, epi, phi, z_evt, tau)
        slowness1 = sqrt((tau - np.abs(z_evt)*ba)**2 - epi**2/alpha**2) / epi
        slowness2 = sqrt(epi**2/beta**2 - dis**2/alpha**2) / np.abs(z_evt)
        brvSc = quadl(brSct, slowness1, slowness2)

        brSdt = lambda q: brSd(q, epi, phi, z_evt, tau)
        slowness1 = 0.0
        slowness2 = sqrt((tau-np.abs(z_evt)*ba)**2 - epi**2/alpha**2) / epi
        brvSd = quadl(brSdt, slowness1, slowness2)

        brSet = lambda q: brSe(q, epi, phi, z_evt, tau)
        slowness1 = sqrt(tau**2/dis**2 - 1.0/beta**2) + off
        slowness2 = sqrt((tau-np.abs(z_evt)*ba)**2 - epi**2/alpha**2) / epi
        brvSe = quadl(brSet, slowness1, slowness2)

        brSft = lambda q: brSf(q, epi, phi, z_evt, tau)
        slowness1 = sqrt(epi**2/beta**2 - dis**2/alpha**2) / np.abs(z_evt)
        slowness2 = (tau - np.abs(z_evt)/beta) / epi
        brvSf = quadl(brSft, slowness1, slowness2)

        brSgt = lambda q: brSg(q, epi, phi, z_evt, tau)
        slowness1 = sqrt(tau**2/dis**2 - 1.0/beta**2) + 2.0*off
        slowness2 = (tau - np.abs(z_evt)/beta) / epi
        brvSg = quadl(brSgt, slowness1, slowness2)

        brvS[k,:] = np.real(brvSa+brvSb+brvSc+brvSd+brvSe+brvSf+brvSg)

    # take time derivatives
    ahP = np.gradient(hvP, dt, edge_order=2, axis=0)
    ahP = np.gradient(ahP, dt, edge_order=2, axis=0)
    ahS = np.gradient(hvS, dt, edge_order=2, axis=0)
    ahS = np.gradient(ahS, dt, edge_order=2, axis=0)
    ahI = np.gradient(hvI, dt, edge_order=2, axis=0)
    ahI = np.gradient(ahI, dt, edge_order=2, axis=0)

    abrP = np.gradient(brvP, dt, edge_order=2, axis=0)
    abrP = np.gradient(abrP, dt, edge_order=2, axis=0)
    abrS = np.gradient(brvS, dt, edge_order=2, axis=0)
    abrS = np.gradient(abrS, dt, edge_order=2, axis=0)

    greens = ahP + ahS + ahI - abrP - abrS

    # convolution with source time function
    dg_half = np.zeros((time.size, 3))
    dg_full = np.zeros((time.size, 3))

    for k in range(time.size):
        dg_half[k,:] = sum(greens[:k+1])*dt
        dg_full[k,:] = sum(ahI[:k+1])*dt

    return dg_full, dg_half


# !------------------!
# !  set parameters  !
# !------------------!

# degrees <--> radians
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

# set Earth radius
rt = 6371e3

# set gravitational constant (SI)
G = 6.67e-11

# set P-wave velocity (M/S)
alpha = 7800.0

# set S-wave velocity (M/S)
beta = 4400.0

# set sampling rate
fs = 10.0


# !------------------------!
# !  set event parameters  !
# !------------------------!

# set file name
file = './sources.dat'

# set keyword arguments
kwargs = dict(skiprows=1, unpack=True)

# read event names
evt_names = np.loadtxt(file, usecols=0, dtype=np.str, **kwargs)

# read moment tensors
mrr, mtt, mpp, mrt, mrp, mtp = np.loadtxt(file, usecols=range(1,7), **kwargs)

# read event coordinates
evt_lat, evt_lon, evt_dep = np.loadtxt(file, usecols=(7, 8, 9), **kwargs)

# (km, down) --> (m, up)
evt_dep = - evt_dep * 1000.0


# !--------------------------!
# !  set station parameters  !
# !--------------------------!

# set file name
file = './receivers_tp.dat'

# set keyword arguments
kwargs = dict(skiprows=1, unpack=True)

# read station names
sta_names = np.loadtxt(file, usecols=0, dtype=np.str, **kwargs)

# read station coordinates
sta_lat, sta_lon, sta_dep = np.loadtxt(file, usecols=(1, 2, 3), **kwargs)


# !----------------------------------!
# !  dispatch computation over cpus  !
# !----------------------------------!

sub_names = sta_names[rank::size]

sub_lon = sta_lon[rank::size]
sub_lat = sta_lat[rank::size]
sub_dep = sta_dep[rank::size]


# !-------------------------------------------!
# !  computation of the analytical solutions  !
# !-------------------------------------------!

for i, sta_name in enumerate(sub_names):
    for j, evt_name in enumerate(evt_names):

        # !-------------------------------------!
        # !  computation of P-wave travel time  !
        # !-------------------------------------!

        # spherical coordinates for epicenter and station
        r0 = rt - evt_dep[j]
        r1 = rt - sub_dep[i]

        t0 = (90.0 - evt_lat[j]) * deg2rad
        t1 = (90.0 - sub_lat[i]) * deg2rad

        p0 = evt_lon[j] * deg2rad
        p1 = sub_lon[i] * deg2rad

        # cartesian coordinates for epicenter and station
        x0 = r0*np.sin(t0)*np.cos(p0)
        y0 = r0*np.sin(t0)*np.sin(p0)
        z0 = r0*np.cos(t0)

        x1 = r1*np.sin(t1)*np.cos(p1)
        y1 = r1*np.sin(t1)*np.sin(p1)
        z1 = r1*np.cos(t1)

        distance = sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

        # arrival_time = int(distance / alpha)
        arrival_time = np.around(distance/alpha, decimals=1)

        # set time vector
        time = np.linspace(0.0, arrival_time, int(arrival_time*fs)+1)


        # !-------------------------------------!
        # !  computation of the gravity strain  !
        # !-------------------------------------!

        print 'rank {0:03d}: {1:} at {2:} ...'.format(rank, evt_name, sta_name)

        # 10m-away stations for computation of gradient
        dphi = 5.0 / (rt*np.sin((90.0-sub_lat[i])*deg2rad)) * rad2deg
        dtht = 5.0 / rt * rad2deg

        # compute prompt gravity perturbations
        _, dg_half_e = harms(evt_lat[j], evt_lon[j], evt_dep[j],
                             sub_lat[i], sub_lon[i]+dphi, sub_dep[i],
                             mrr[j], mtt[j], mpp[j], mrt[j], mrp[j], mtp[j],
                             time, 1.0/fs)

        _, dg_half_w = harms(evt_lat[j], evt_lon[j], evt_dep[j],
                             sub_lat[i], sub_lon[i]-dphi, sub_dep[i],
                             mrr[j], mtt[j], mpp[j],
                             mrt[j], mrp[j], mtp[j],
                             time, 1.0/fs)

        _, dg_half_n = harms(evt_lat[j], evt_lon[j], evt_dep[j],
                             sub_lat[i]+dtht, sub_lon[i], sub_dep[i],
                             mrr[j], mtt[j], mpp[j], mrt[j], mrp[j], mtp[j],
                             time, 1.0/fs)

        _, dg_half_s = harms(evt_lat[j], evt_lon[j], evt_dep[j],
                             sub_lat[i]-dtht, sub_lon[i], sub_dep[i],
                             mrr[j], mtt[j], mpp[j], mrt[j], mrp[j], mtp[j],
                             time, 1.0/fs)

        # get gravity gradient
        h_ee = (dg_half_e[:, 0] - dg_half_w[:, 0]) / 10.0
        h_en = (dg_half_e[:, 1] - dg_half_w[:, 1]) / 10.0
        h_ez = (dg_half_e[:, 2] - dg_half_w[:, 2]) / 10.0

        h_ne = (dg_half_n[:, 0] - dg_half_s[:, 0]) / 10.0
        h_nn = (dg_half_n[:, 1] - dg_half_s[:, 1]) / 10.0
        h_nz = (dg_half_n[:, 2] - dg_half_s[:, 2]) / 10.0

        # get gravity strain
        h_ee = cumtrapz(h_ee, dx=1.0/fs, initial=0)
        h_ee = cumtrapz(h_ee, dx=1.0/fs, initial=0)

        h_en = cumtrapz(h_en, dx=1.0/fs, initial=0)
        h_en = cumtrapz(h_en, dx=1.0/fs, initial=0)

        h_ez = cumtrapz(h_ez, dx=1.0/fs, initial=0)
        h_ez = cumtrapz(h_ez, dx=1.0/fs, initial=0)

        h_ne = cumtrapz(h_ne, dx=1.0/fs, initial=0)
        h_ne = cumtrapz(h_ne, dx=1.0/fs, initial=0)

        h_nn = cumtrapz(h_nn, dx=1.0/fs, initial=0)
        h_nn = cumtrapz(h_nn, dx=1.0/fs, initial=0)

        h_nz = cumtrapz(h_nz, dx=1.0/fs, initial=0)
        h_nz = cumtrapz(h_nz, dx=1.0/fs, initial=0)


        # !----------------------!
        # !  save the solutions  !
        # !----------------------!

        file = './grstrain_' + sta_name + '_' + evt_name + '.txt'

        np.savetxt(file, np.c_[time, h_ee, h_en, h_ne, h_nn, h_ez, h_nz],
                   fmt='%.2f %.12e %.12e %.12e %.12e %.12e %.12e')

