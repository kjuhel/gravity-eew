#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute optimal signal-to-noise ratio
"""

import numpy as np

from obspy.geodetics import gps2dist_azimuth
from scipy.signal import iirfilter, sosfilt

from mpi4py import MPI


# !-----------------!
# !  set functions  !
# !-----------------!

def stf_quadra(m0, time, half_duration):
    """ compute moment rate STF, for a given moment M0 and half-duration """

    # constant such that int(dm0) = m0
    a = 1.0 / (1.0/3.0 + 1.0/13.0 - 6.0/11.0 + 5.0/3.0 - 20.0/7.0 + 2.0)

    duration = 2.0*half_duration

    mask1 = (time >= 0) * (time <= half_duration)
    mask2 = (time > half_duration) * (time <= duration)

    moment_rate = np.zeros_like(time, dtype=np.float128)

    moment_rate[mask1] = (time[mask1] / half_duration)**2
    moment_rate[mask2] = (1.0 - (time[mask2]/half_duration-1.0)**2)**6

    moment_rate = moment_rate * a*m0 / half_duration

    return moment_rate


def rotate(h, theta):
    """ rotation of coordinates
        input h  : [xx, xy, yx, yy, xz, yz]
        output h : [rr, rt, tr, tt, tz, rz]
        theta: azimuth (in radians)
    """

    h_rot = np.zeros_like(h)

    h_rot[0] = +h[0]*(np.cos(theta))**2 + h[3]*(np.sin(theta))**2
    h_rot[0] = h_rot[0] + (h[1]+h[2])*np.cos(theta)*np.sin(theta)

    h_rot[1] = +h[1]*(np.cos(theta))**2 - h[2]*(np.sin(theta))**2
    h_rot[1] = h_rot[1] - (h[0]-h[3])*np.cos(theta)*np.sin(theta)

    h_rot[2] = -h[1]*(np.sin(theta))**2 + h[2]*(np.cos(theta))**2
    h_rot[2] = h_rot[2] - (h[0]-h[3])*np.cos(theta)*np.sin(theta)

    h_rot[3] = +h[0]*(np.sin(theta))**2 + h[3]*(np.cos(theta))**2
    h_rot[3] = h_rot[3] - (h[1]+h[2])*np.cos(theta)*np.sin(theta)

    h_rot[4] = +h[4]*np.cos(theta) + h[5]*np.sin(theta)
    h_rot[5] = -h[4]*np.sin(theta) + h[5]*np.cos(theta)

    return h_rot


def noise_toba(noisefloor, cutoff, fs, n_fft):
    """ compute TOBA noise time-series, for given
    noise-floor and cut-off frequency """

    ff = np.fft.fftfreq(n_fft, 1.0/fs)

    shotnoise = noisefloor
    seismicnoise = noisefloor / (ff[1:]/cutoff)**2

    spec_instrnoise = np.zeros(n_fft)

    spec_instrnoise[1:] = shotnoise + seismicnoise
    spec_instrnoise[0] = spec_instrnoise[1]

    whitenoise = np.sqrt(0.5*fs) * np.random.randn(n_fft)
    whitenoise = whitenoise - whitenoise.mean()

    spec_whitenoise = np.fft.fft(whitenoise)

    spec_tobanoise = spec_instrnoise * np.conjugate(spec_whitenoise)
    tobanoise = np.real(np.fft.ifft(spec_tobanoise))

    return tobanoise


# !-----------------------!
# !  set some parameters  !
# !-----------------------!

# set path
path = '../data/TP/half/'

# degrees --> radians
deg2rad = np.pi / 180.0

# sampling rate (Hz)
fs = 10.0

# noise-floor and cut-off frequency
noisefloors = [1.0e-15, 1.0e-15, 1.0e-14, 5.0e-17]
cutoffs = [0.05, 0.10, 0.05, 0.50]

# set number of models
n_model = len(noisefloors)

# number of channels
n_channel = 5  # a*XX + b*YY + c*XY = PP,
               # d*XX + e*YY + f*XY = CC,
               # ZZ, XZ, YZ


# !-----------------------------!
# !  set earthquake parameters  !
# !-----------------------------!

# set event location
evt_lat, evt_lon, evt_dep = 0.0, 0.0, 20e3

# set event names
evt_names = ['VRTSTRIKESLIP', 'RVRSDIPSLIP10', 'RVRSDIPSLIP20']

n_event = len(evt_names)


# !--------------------------!
# !  set station parameters  !
# !--------------------------!

# set file name
file = './dat_files/receivers_tp.dat'

# set keyword arguments
kwargs = dict(skiprows=1, unpack=True)

# read station names
sta_names = np.loadtxt(file, usecols=0, dtype=np.str, **kwargs)

# read file coordinates
latitudes, longitudes = np.loadtxt(file, usecols=(1, 2), **kwargs)

n_station = len(sta_names)


# !-------------------------------!
# !  computation of STF database  !
# !-------------------------------!

# set magnitude vector
magnitudes = np.linspace(5.0, 9.1, 101)
n_magnitude = len(magnitudes)

moment_rates = []

for magnitude in magnitudes:

    # set total seismic moment
    m0 = 10**(1.5 * (magnitude + 6.066666))

    # set corresponding half duration
    half_duration = 0.5 * (m0 / 1.0e16)**(1.0/3.0)
    half_duration = np.around(half_duration, decimals=1)

    duration = 2.0 * half_duration

    # set time vector
    time = np.linspace(0.0, duration, int(duration*fs)+1)

    # compute moment-rate source time function
    moment_rate = stf_quadra(m0, time, half_duration)

    moment_rates.append(moment_rate / 7.16e21)


# !----------------------------------!
# !  dispatch computation over cpus  !
# !----------------------------------!

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

sub_names = sta_names[rank::size]

sub_latitudes = latitudes[rank::size]
sub_longitudes = longitudes[rank::size]

for i, sta_name in enumerate(sub_names):
    print 'rank {0:02d} handles grid point {1:}'.format(rank, sta_name)

    sta_lat, sta_lon = sub_latitudes[i], sub_longitudes[i]

    _, faz, _ = gps2dist_azimuth(evt_lat, evt_lon, sta_lat, sta_lon,
                                 a=6371e3, f=0.0)

    azimuth = (90.0 - faz) * deg2rad

    for j, evt_name in enumerate(evt_names):

        # !--------------------------------!
        # !  read gravity strain template  !
        # !--------------------------------!

        # set file name
        file = f'{path}grstrain/grstrain_{sta_name}_{evt_name}.txt'

        # read gravity strain template
        time, ee, en, ne, nn, ez, nz = np.loadtxt(file, unpack=True)

        # input h : [xx, xy, yx, yy, xz, yz]
        h_zne = np.array([ee, en, ne, nn, ez, nz])

        # output h : [rr, rt, tr, tt, rz, tz]
        h_zrt = rotate(h_zne, azimuth)

        h = np.zeros((n_channel, time.size))

        h[0] = 0.5 * (h_zrt[0] - h_zrt[3])  # PP
        h[1] = 0.5 * (h_zrt[1] + h_zrt[2])  # CC
        h[2] = - (h_zne[3] + h_zne[0])      # ZZ
        h[3] = h_zrt[4]                     # RZ
        h[4] = h_zrt[5]                     # TZ

        # initialize SNR array
        snr = np.zeros((n_model, n_channel, n_magnitude))

        for mod in range(n_model):

            # set noise floor and cut-off frequency
            noisefloor, cutoff = noisefloors[mod], cutoffs[mod]

            # set whitening filter
            kwargs = dict(ftype='butter', btype='high', output='sos')
            sos = iirfilter(2, cutoff/(0.5*fs), **kwargs)

            for mag in range(n_magnitude):

                # set moment-rate STF
                stf = moment_rates[mag]

                for cha in range(n_channel):

                    # convolve by STF
                    template = np.convolve(h[cha], stf) / fs

                    # apply whitening filter
                    template = sosfilt(sos, template)
                    template = template[:time.size]

                    # energy of the signal = template auto-correlation
                    E = np.correlate(template, template) / fs

                    # noise (power) spectral density
                    N = noisefloor**2

                    # signal-to-noise ratio
                    snr[mod, cha, mag] = np.sqrt(2.0*E/N)

        # save results
        file = f'{path}SNR_optimal/snr_tp_{sta_name}_{evt_name}.npy'
        np.save(file, snr)
