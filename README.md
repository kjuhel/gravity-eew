# G-EEW
Earthquake early detection capabilities of different types of
future-generation gravity gradiometers.

This repository contains codes used in [Shimoda et al. (2021)](https://doi.org/10.1093/gji/ggaa486).


## Simulation and data processing

### Computation of prompt gravity strain

The perturbation of the gravitational field is computed based on the half-space model
developed by [Harms (2016)](https://doi.org/10.1093/gji/ggw076).
Then each component of the gravity gradient is approximated numerically as the
finite difference of the gravity perturbations at two closely located points.
Gravity strain `h(t)` is obtained by integrating the gravity gradient twice over time.


### Computation of the optimal signal-to-noise ratio

We consider an optimal matched-filter detection procedure.
The optimal matched-filters are the pre-whitened signal templates `h(t)`.
Both the signal templates and the recorded data `s(t) = h(t) + n(t)`, where `n(t)` is detector noise,
are whitened by deconvolving them by the power spectrum of the detector noise.

The matched-filter output is obtained by correlating the whitened template with the whitened data.
The signal-to-noise ratio (SNR) is defined as the ratio between the matched-filter output and
the standard deviation of the matched-filter applied to noise alone.


## Contact information
* [Kévin Juhel](mailto:juhel.kevin@gmail.com)
* [Tomofumi Shimoda](mailto:shimoda@granite.phys.s.u-tokyo.ac.jp)
* [Jean-Paul Ampuero](mailto:ampuero@geoazur.unice.fr)


## References
* Shimoda, T., Juhel, K., Ampuero, J. P., Montagner, J. P., & Barsuglia, M. (2021).
Early earthquake detection capabilities of different types of future-generation gravity gradiometers.
Geophysical Journal International, 224(1), 533-542.
* Harms, J. (2016). Transient gravity perturbations from a double-couple in a homogeneous half-space.
Geophysical Journal International, 205(2), 1153-1164.
