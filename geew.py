import numpy as np
import scipy

from scipy import signal
from obspy.geodetics import gps2dist_azimuth


#  !------------------------!
#  !   set some functions   !
#  !------------------------!

def quad_routine(func, a, b, x_list, w_list):
    c_1 = (b-a)/2.0
    c_2 = (b+a)/2.0
    eval_points = map(lambda x: c_1*x+c_2, x_list)
    func_evals = map(func, eval_points)
    func_evals = list(map(list, zip(*func_evals)))
    return c_1 * sum(scipy.array(func_evals)[0] * scipy.array(w_list)), c_1 * sum(scipy.array(func_evals)[1] * scipy.array(w_list)), c_1 * sum(scipy.array(func_evals)[2] * scipy.array(w_list))


def quadgk(func, a, b):
    x_kr = [-0.991455371120813, -0.949107912342759, -0.864864423359769,
            -0.741531185599394, -0.586087235467691,-0.405845151377397,
            -0.207784955007898, 0.0, 0.207784955007898,0.405845151377397,
            0.586087235467691, 0.741531185599394, 0.864864423359769,
            0.949107912342759, 0.991455371120813]

    w_kr = [0.022935322010529, 0.063092092629979, 0.104790010322250,
            0.140653259715525, 0.169004726639267, 0.190350578064785,
            0.204432940075298, 0.209482141084728, 0.204432940075298,
            0.190350578064785, 0.169004726639267, 0.140653259715525,
            0.104790010322250, 0.063092092629979, 0.022935322010529]
    return np.array(quad_routine(func, a, b, x_kr, w_kr))


def quadl(func, a, b):
    x_lo =  [-1.0, -0.965245926503838572795851392070,
             -0.885082044222976298825401631482,
             -0.763519689951815200704118475976,
             -0.606253205469845711123529938637,
             -0.420638054713672480921896938739,
             -0.215353955363794238225679446273, 0.0,
             0.215353955363794238225679446273,
             0.420638054713672480921896938739,
             0.606253205469845711123529938637,
             0.763519689951815200704118475976,
             0.885082044222976298825401631482,
             0.965245926503838572795851392070, 1.0]

    w_lo = [0.952380952380952380952380952381E-02,
            0.580298930286012490968805840253E-01,
            0.101660070325718067603666170789, 0.140511699802428109460446805644,
            0.172789647253600949052077099408, 0.196987235964613356092500346507,
            0.211973585926820920127430076977, 0.217048116348815649514950214251,
            0.211973585926820920127430076977, 0.196987235964613356092500346507,
            0.172789647253600949052077099408, 0.140511699802428109460446805644,
            0.101660070325718067603666170789,
            0.580298930286012490968805840253E-01, 0.952380952380952380952380952381E-02]
    return np.array(quad_routine(func, a, b, x_lo, w_lo))


def analytic_halfspace(evt_latitude, evt_longitude, evt_depth,
                       sta_latitude, sta_longitude, sta_depth,
                       Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
                       time_vector, sampling_rate):

    """Compute prompt gravity perturbations from a
    double-couple buried in a homogeneous half-space.
    Corresponding publication: Harms (GJI, 2016).
    Developed in Matlab by J. Harms.
    Translated into Python by K. Juhel"""

    # x = East, y = North, z = Upward

    #  !--------------------!
    #  !   set parameters   !
    #  !--------------------!

    G = 6.6723e-11
    rt = 6371000.

    alpha, beta = 7800., 4400.

    # degrees <--> radians
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    depth = evt_depth


    #  !------------------------------------------------------------!
    #  !   compute azimuth and distance from epicenter to station   !
    #  !------------------------------------------------------------!

    # compute forward azimuth
    [r, faz, baz] = gps2dist_azimuth(evt_latitude, evt_longitude,
                                     sta_latitude, sta_longitude,
                                     a=rt, f=0.0)

    phi = (90.0 - faz) * deg2rad

    # compute distance from epicenter to station
    R = np.sqrt(r**2 + depth**2)

    #  !-----------------------!
    #  !   set moment tensor   !
    #  !-----------------------!

    M = np.array([[Mpp, -Mtp, Mrp], [-Mtp, Mtt, -Mrt], [Mrp, -Mrt, Mrr]])


    #  !------------------------!
    #  !   computation time !   !
    #  !------------------------!

    # gravity perturbation in transform domain (DoubleCouple_HalfSpacenb)
    # include possibility for sign flip for branch-cut navigation
    eta_i = lambda p, q, si: si*scipy.sqrt(q**2 - p**2)
    eta_a = lambda p, q, sa: sa*scipy.sqrt(1.0/alpha**2 + q**2 - p**2)
    eta_b = lambda p, q: scipy.sqrt(1.0/beta**2 + q**2 - p**2)

    Rp = lambda p, q, sa: (eta_b(p, q)**2 + (q**2-p**2))**2 - 4.0*(q**2-p**2)*eta_a(p, q, sa)*eta_b(p, q)

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

    nI = lambda p, q, phi, si: np.array([xI(M,p,q,phi,si), yI(M,p,q,phi,si), zI(M,p,q,phi,si)])

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

    nP = lambda p, q, phi, si, sa: np.array([xP(M,p,q,phi,si,sa), yP(M,p,q,phi,si,sa), zP(M,p,q,phi,si,sa)])

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

    nS = lambda p, q, phi, si, sa: np.array([xS(M,p,q,phi,si,sa), yS(M,p,q,phi,si,sa), zS(M,p,q,phi,si,sa)])

    #  !------------------------!
    #  !   Cagniard - de Hoop   !
    #  !------------------------!

    # alpha, beta, inf. branch points
    pb1 = lambda q: q
    pb2 = lambda q: scipy.sqrt(1.0/alpha**2 + q**2)
    pb3 = lambda q: scipy.sqrt(1.0/beta**2 + q**2)

    # Cagniard paths
    p1 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*scipy.sqrt(t**2-(pb1(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)
    p2 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*scipy.sqrt(t**2-(pb2(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)
    p3 = lambda r, z, t, q: (r*t+1.0j*np.abs(z)*scipy.sqrt(t**2-(pb3(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)

    dpdt1 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/scipy.sqrt(t**2-(pb1(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)
    dpdt2 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/scipy.sqrt(t**2-(pb2(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)
    dpdt3 = lambda r, z, t, q: (r+1.0j*t*np.abs(z)/scipy.sqrt(t**2-(pb3(q)*scipy.sqrt(r**2+z**2))**2))/(r**2+z**2)

    #  !----------------------------------------------!
    #  !   routines : contributions from hyperbolae   !
    #  !----------------------------------------------!

    def hI(q, r, phi, z, t):
        bool = 1.0
        diff = nI(p1(r,z,t,q),q,phi,1)*dpdt1(r,z,t,q) - nI(np.conj(p1(r,z,t,q)),q,phi,1)*np.conj(dpdt1(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2

    def hP(q, r, phi, z, t):
        bool = (t > scipy.sqrt(r**2+z**2)/alpha)
        diff = nP(p2(r,z,t,q),q,phi,1,1)*dpdt2(r,z,t,q) - nP(np.conj(p2(r,z,t,q)),q,phi,1,1)*np.conj(dpdt2(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2

    def hS(q, r, phi, z, t):
        bool = (t > scipy.sqrt(r**2+z**2)/beta)
        diff = nS(p3(r,z,t,q),q,phi,1,1)*dpdt3(r,z,t,q) - nS(np.conj(p3(r,z,t,q)),q,phi,1,1)*np.conj(dpdt3(r,z,t,q))
        return bool * np.real(diff) / (2.0*np.pi)**2

    #  !---------------------------------------------------------!
    #  !   routines : contributions from branch-cut navigation   !
    #  !---------------------------------------------------------!

    ba = scipy.sqrt(1.0/beta**2-1.0/alpha**2)

    def brPa(q, r, phi, z, t):
        bool = (t > np.abs(z)/alpha) * (t < scipy.sqrt(r**2+z**2)/alpha)
        diff = nP(p2(r,z,t,q), q, phi, 1, 1) - nP(p2(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt2(r,z,t,q)) / (2.0*np.pi)**2

    def brPb(q, r, phi, z, t):
        bool = (t > scipy.sqrt(r**2+z**2)/alpha) * (t < (r**2+z**2)/(alpha*np.abs(z)))
        diff = nP(p2(r,z,t,q), q, phi, 1, 1) - nP(p2(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt2(r,z,t,q)) / (2.0*np.pi)**2

    def brSa(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t > np.abs(z)/beta)
        bool = bool * (t < r/alpha+ba*np.abs(z))
        diff = (nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1))
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSb(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t > r/alpha+ba*np.abs(z))
        bool = bool * (t < np.abs(z)/beta + r/np.abs(z)*scipy.sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSc(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t < (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t > np.abs(z)/beta+r/np.abs(z)*scipy.sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSd(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t > r/alpha+ba*np.abs(z))
        bool = bool * (t < scipy.sqrt(r**2+z**2)/beta)
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, -1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSe(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t > scipy.sqrt(r**2+z**2)/beta)
        bool = bool * (t < (r**2+z**2)/np.abs(z)*ba)
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, -1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSf(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t < (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t > np.abs(z)/beta+r/np.abs(z)*scipy.sqrt(r**2/beta**2-(r**2+z**2)/alpha**2))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    def brSg(q, r, phi, z, t):
        bool = (r/scipy.sqrt(r**2+z**2) > beta/alpha) * (t > (r**2+z**2)/np.abs(z)*ba)
        bool = bool * (t < (r**2+z**2)/(beta*np.abs(z)))
        diff = nS(p3(r,z,t,q), q, phi, 1, 1) - nS(p3(r,z,t,q), q, phi, -1, 1)
        return np.real(bool * diff * dpdt3(r,z,t,q)) / (2.0*np.pi)**2

    hvI = np.zeros( ( len(time_vector), 3 ) )
    hvP = np.zeros( ( len(time_vector), 3 ) )
    hvS = np.zeros( ( len(time_vector), 3 ) )

    off = 1.0e-6

    for k in range( len(time_vector) ):

        tau = time_vector[k]

        # contributions from hyperbolae
        hIt = lambda q: hI(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = tau/R - off
        hvI[k,:] = np.real(quadgk(hIt, slowness1, slowness2))

        hPt = lambda q: hP(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = scipy.sqrt((tau/R)**2 - 1.0/alpha**2) - off
        hvP[k,:] = np.real(quadgk(hPt, slowness1, slowness2))

        hSt = lambda q: hS(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = scipy.sqrt((tau/R)**2 - 1.0/beta**2) - off
        hvS[k,:] = np.real(quadgk(hSt, slowness1, slowness2))


    brvP, brvS = np.zeros( ( len(time_vector), 3 ) ), np.zeros( ( len(time_vector), 3 ) )

    for k in range( len(time_vector) ):

        tau = time_vector[k]

        # contributions from branch-cut navigation
        brPat = lambda q: brPa(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = (tau - np.abs(depth)/alpha) / r
        brvPa = quadgk(brPat, slowness1, slowness2)

        brPbt = lambda q: brPb(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt((tau/R)**2 - 1.0/alpha**2)
        slowness2 = (tau - np.abs(depth)/alpha) / r
        brvPb = quadgk(brPbt, slowness1, slowness2)

        brvP[k,:] = np.real(brvPa + brvPb)

        brSat = lambda q: brSa(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = (tau - np.abs(depth)/beta) / r
        brvSa = quadl(brSat, slowness1, slowness2)

        brSbt = lambda q: brSb(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt((tau-np.abs(depth)*ba)**2 - r**2/alpha**2) / r
        slowness2 = (tau - np.abs(depth)/beta) / r
        brvSb = quadl(brSbt, slowness1, slowness2)

        brSct = lambda q: brSc(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt((tau - np.abs(depth)*ba)**2 - r**2/alpha**2) / r
        slowness2 = scipy.sqrt(r**2/beta**2 - R**2/alpha**2) / np.abs(depth)
        brvSc = quadl(brSct, slowness1, slowness2)

        brSdt = lambda q: brSd(q, r, phi, depth, tau)
        slowness1 = 0.0
        slowness2 = scipy.sqrt((tau-np.abs(depth)*ba)**2 - r**2/alpha**2) / r
        brvSd = quadl(brSdt, slowness1, slowness2)

        brSet = lambda q: brSe(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt(tau**2/R**2 - 1.0/beta**2) + off
        slowness2 = scipy.sqrt((tau-np.abs(depth)*ba)**2 - r**2/alpha**2) / r
        brvSe = quadl(brSet, slowness1, slowness2)

        brSft = lambda q: brSf(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt(r**2/beta**2 - R**2/alpha**2) / np.abs(depth)
        slowness2 = (tau - np.abs(depth)/beta) / r
        brvSf = quadl(brSft, slowness1, slowness2)

        brSgt = lambda q: brSg(q, r, phi, depth, tau)
        slowness1 = scipy.sqrt(tau**2/R**2 - 1.0/beta**2) + 2.0*off
        slowness2 = (tau - np.abs(depth)/beta) / r
        brvSg = quadl(brSgt, slowness1, slowness2)

        brvS[k,:] = np.real(brvSa+brvSb+brvSc+brvSd+brvSe+brvSf+brvSg)


    # take time derivatives
    dt = 1. / sampling_rate
    dx = dt

    ahP = np.gradient(np.gradient(hvP, dx, axis=0), dx, axis=0)
    ahS = np.gradient(np.gradient(hvS, dx, axis=0), dx, axis=0)
    ahI = np.gradient(np.gradient(hvI, dx, axis=0), dx, axis=0)

    abrP = np.gradient(np.gradient(brvP, dx, axis=0), dx, axis=0)
    abrS = np.gradient(np.gradient(brvS, dx, axis=0), dx, axis=0)


    greens = ahP + ahS + ahI - abrP - abrS

    # convolution with source time function
    dg = np.zeros( ( len(time_vector), 3 ) )

    for k in range( len(time_vector) ):
        dg[k,:] = sum(greens[:k+1])*dt

    return dg


def source_template_linear(total_moment, tt, half_duration):
    """ computes source template, for a given moment M0 and time axis """

    duration = 2.0*half_duration

    mask1 = (tt >= 0) * (tt <= half_duration)
    mask2 = (tt > half_duration) * (tt <= duration)

    moment_rate = np.zeros( len(tt) )

    moment_rate[mask1] = +tt[mask1] / half_duration**2
    moment_rate[mask2] = -tt[mask2] / half_duration**2 + 2.0 / half_duration

    moment_rate = moment_rate * total_moment

    return moment_rate


def source_template_quadra(total_moment, tt, half_duration):
    """ computes source template, for a given moment M0 and time axis """

    a = 1.0 / (1.0/3.0 + 1.0/13.0 - 6.0/11.0 + 5.0/3.0 - 20.0/7.0 + 2.0)

    duration = 2.0*half_duration

    mask1 = (tt >= 0) * (tt <= half_duration)
    mask2 = (tt > half_duration) * (tt <= duration)

    moment_rate = np.zeros( len(tt) )

    moment_rate[mask1] = ( tt[mask1] / half_duration )**2
    moment_rate[mask2] = ( 1.0 - (tt[mask2]/half_duration-1.0)**2 )**6

    moment_rate = moment_rate * a * total_moment / half_duration

    return moment_rate


def noise_toba(noisefloor, cutoff, sampling_rate, length):
    """ computes TOBA noise time series, for
    given noise-floor and cut-off frequency"""

    ff = np.fft.fftfreq(length, 1.0 / sampling_rate)

    shotnoise    = noisefloor
    seismicnoise = noisefloor / (ff[1:]/cutoff)**2

    spec_instrnoise = np.zeros(length)

    spec_instrnoise[1:] = shotnoise + seismicnoise
    spec_instrnoise[0] = spec_instrnoise[1]

    whitenoise = np.sqrt(0.5*sampling_rate) * np.random.randn(length)
    spec_whitenoise = np.fft.fft(whitenoise)

    spec_tobanoise = spec_instrnoise * np.conjugate(spec_whitenoise)
    tobanoise = np.real( np.fft.ifft(spec_tobanoise) )

    return tobanoise


#   !-------------------------!
#   !   set some parameters   !
#   !-------------------------!

# degrees <--> radians
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

# Earth radius
rt = 6371000.0

# P-wave velocity
alpha = 7800.0

# set sampling rate
sampling_rate = 10.0


#   !--------------------------------------!
#   !   set gravity strainmeters network   !
#   !--------------------------------------!

# number of gravity strainmeters
number_stations = 6

# set network configuration
sta_names = ['RCVR1', 'RCVR2', 'RCVR3', 'RCVR4', 'RCVR5', 'RCVR6']

sta_latitudes  = [38.50, 37.75, 37.00, 45.00, 45.00, 43.70]
sta_longitudes = [140.5, 139.5, 140.5, 135.3, 134.0, 134.0]

sta_depth = 0.


#   !-------------------------------!
#   !   set earthquake parameters   !
#   !-------------------------------!

# set event name
evt_name = 'TOHOKU'


if evt_name == 'TOHOKU':

    # set epicenter location
    evt_latitude, evt_longitude = 37.52, 143.05
    evt_colatitude = 90.0 - evt_latitude

    # set epicenter depth
    evt_depth = 20000.0

    # set seismic moment tensor (global CMT solution)
    mrr, mtt, mpp = +1.73000e+22, -0.28100e+22, -1.45000e+22
    mrt, mrp, mtp = +2.12000e+22, +4.55000e+22, -0.65700e+22

    # set STF half duration
    half_duration = 70.0


elif evt_name == 'FORESHOCK':

    # set epicenter location
    evt_latitude, evt_longitude = 38.56, 142.78
    evt_colatitude = 90.0 - evt_latitude

    # set seismic moment tensor (global CMT solution)
    mrr, mtt, mpp = +0.489e+20, -0.024e+20, -0.465e+20
    mrt, mrp, mtp = +0.418e+20, +1.050e+20, -0.128e+20

    # set STF half duration
    half_duration = 11.3

    # set epicenter depth
    evt_depth = 14100.0


#  !-----------------------------------------------!
#  !   computation of maximum P-wave travel time   !
#  !-----------------------------------------------!

tp_min, tp_max = 1000.0, 0.0

for i in range(number_stations):

    sta_latitude, sta_longitude = sta_latitudes[i], sta_longitudes[i]
    sta_colatitude = 90.0 - sta_latitude

    sta_name = sta_names[i]

    # spherical coordinates for epicenter and station
    r0, t0, p0 = rt - evt_depth, evt_colatitude * deg2rad, evt_longitude * deg2rad
    r1, t1, p1 = rt - sta_depth, sta_colatitude * deg2rad, sta_longitude * deg2rad

    # cartesian coordinates for epicenter and station
    x0, y0, z0 = r0*np.sin(t0)*np.cos(p0), r0*np.sin(t0)*np.sin(p0), r0*np.cos(t0)
    x1, y1, z1 = r1*np.sin(t1)*np.cos(p1), r1*np.sin(t1)*np.sin(p1), r1*np.cos(t1)

    distance = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

    arrival_time = distance / alpha
    arrival_time = np.around([arrival_time], decimals=1)[0]

    tp_min = min(tp_min, arrival_time)
    tp_max = max(tp_max, arrival_time)


#  !---------------------------------------------!
#  !   computation of the analytical solutions   !
#  !---------------------------------------------!

for i in range(number_stations):

    sta_latitude, sta_longitude = sta_latitudes[i], sta_longitudes[i]
    sta_colatitude = 90.0 - sta_latitude

    sta_name = sta_names[i]


    #  !---------------------------------------!
    #  !   computation of P-wave travel time   !
    #  !---------------------------------------!

    # spherical coordinates for epicenter and station
    r0, t0, p0 = rt - evt_depth, evt_colatitude * deg2rad, evt_longitude * deg2rad
    r1, t1, p1 = rt - sta_depth, sta_colatitude * deg2rad, sta_longitude * deg2rad

    # cartesian coordinates for epicenter and station
    x0, y0, z0 = r0*np.sin(t0)*np.cos(p0), r0*np.sin(t0)*np.sin(p0), r0*np.cos(t0)
    x1, y1, z1 = r1*np.sin(t1)*np.cos(p1), r1*np.sin(t1)*np.sin(p1), r1*np.cos(t1)

    distance = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

    arrival_time = distance / alpha
    arrival_time = np.around([arrival_time], decimals=1)[0]

    # set time vector
    time_vector = np.linspace(0.0, arrival_time, arrival_time*sampling_rate + 1)


    #  !---------------------------------------------!
    #  !   computation of the analytical solutions   !
    #  !---------------------------------------------!

    print 'computing analytical solutions for event {0:} at {1:} ...'.format(evt_name, sta_name)

    # 10m-away stations for computation of gradient
    dphi = 5. / ( rt * np.sin(sta_colatitude*deg2rad) ) * rad2deg
    dtht = 5. / rt * rad2deg

    # 10m-away stations for computation of gradient
    dphi = 5. / ( rt * np.sin(sta_colatitude*deg2rad) ) * rad2deg
    dtht = 5. / rt * rad2deg

    # compute gravity perturbations
    g1E = analytic_halfspace(evt_latitude, evt_longitude, evt_depth,
                             sta_latitude, sta_longitude+dphi, sta_depth,
                             mrr, mtt, mpp, mrt, mrp, mtp,
                             time_vector, sampling_rate)

    g1W = analytic_halfspace(evt_latitude, evt_longitude, evt_depth,
                             sta_latitude, sta_longitude-dphi, sta_depth,
                             mrr, mtt, mpp, mrt, mrp, mtp,
                             time_vector, sampling_rate)

    g1N = analytic_halfspace(evt_latitude, evt_longitude, evt_depth,
                             sta_latitude+dtht, sta_longitude+dphi, sta_depth,
                             mrr, mtt, mpp, mrt, mrp, mtp,
                             time_vector, sampling_rate)

    g1S = analytic_halfspace(evt_latitude, evt_longitude, evt_depth,
                             sta_latitude-dtht, sta_longitude, sta_depth,
                             mrr, mtt, mpp, mrt, mrp, mtp,
                             time_vector, sampling_rate)


    # compute gravity gradient (half)
    ggradient_EZ = (g1E[:,2] - g1W[:,2]) / 10.
    ggradient_NZ = (g1N[:,2] - g1S[:,2]) / 10.


    # compute gravity strain (half)
    gstrain_EZ = scipy.integrate.cumtrapz(ggradient_EZ, dx=1./sampling_rate, initial=0)
    gstrain_EZ = scipy.integrate.cumtrapz(gstrain_EZ, dx=1./sampling_rate, initial=0)

    gstrain_NZ = scipy.integrate.cumtrapz(ggradient_NZ, dx=1./sampling_rate, initial=0)
    gstrain_NZ = scipy.integrate.cumtrapz(gstrain_NZ, dx=1./sampling_rate, initial=0)


    #  !------------------------!
    #  !   save the solutions   !
    #  !------------------------!

    file = './grstrain_' + sta_name + '_' + evt_name + '.txt'
    np.savetxt(file, np.c_[time_vector, gstrain_EZ, gstrain_NZ], fmt='%.2f %.12e %.12e')


#  !--------------------------------!
#  !   event source time function   !
#  !--------------------------------!

sampling_rate = 10.0

time = np.linspace(0.0, 160.0, int(160.0*sampling_rate)+1)
moment_rate = source_template_linear(1.0, time, half_duration)


# define filter for pre-whitening
b, a = signal.iirfilter(2, 0.05 / (0.5*sampling_rate), btype='high', ftype='butter')
# b, a = signal.iirfilter(2, 0.10 / (0.5*sampling_rate), btype='high', ftype='butter')
# b, a = signal.iirfilter(2, 0.05 / (0.5*sampling_rate), btype='high', ftype='butter')
# b, a = signal.iirfilter(2, 0.50 / (0.5*sampling_rate), btype='high', ftype='butter')


for i, sta_name in enumerate(sta_names):

    # set gravity strain file names
    green_file = './grstrain_' + sta_name + '_' + evt_name + '.txt'

    # open gravity strain data file
    with open(green_file, 'r') as f:
        content = f.readlines()

        time = np.array([float(line.split()[0]) for line in content])

        greenEZ = np.array([float(line.split()[1]) for line in content])
        greenNZ = np.array([float(line.split()[2]) for line in content])

    green = np.array( [greenEZ, greenNZ] )


    #  !-----------------------!
    #  !   append with zeros   !
    #  !-----------------------!

    pre_time = np.linspace(-50.0, -1.0/sampling_rate, int(50.0*sampling_rate))

    time = np.append(pre_time, time)
    green = np.c_[np.zeros((2, len(pre_time))), green]


    #  !--------------------------!
    #  !   convolution with STF   !
    #  !--------------------------!

    temp = np.array([np.convolve(g, moment_rate) / sampling_rate for g in green])
    temp = temp[:, :len(green[0])]


    #  !------------------------------------!
    #  !   compute TOBA noise time series   !
    #  !------------------------------------!

    nois = noise_toba(1.0e-15, 0.05, sampling_rate, 2*len(time))
    #nois = noise_toba(1.0e-15, 0.10, sampling_rate, 2*len(time))
    #nois = noise_toba(1.0e-14, 0.05, sampling_rate, 2*len(time))
    #nois = noise_toba(5.0e-17, 0.50, sampling_rate, 2*len(time))


    #  !-------------------------!
    #  !   pre-whitening filter  !
    #  !-------------------------!

    nois = signal.lfilter(b, a, nois)
    nois = nois[len(time):]

    temp = signal.lfilter(b, a, temp)


    #  !------------------------------!
    #  !   compute TOBA time series   !
    #  !------------------------------!

    data = nois + 1. * temp
