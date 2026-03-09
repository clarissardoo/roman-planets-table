# Roman Coronagraph Target Planning Pipeline

Generates projected separations, flux contrasts, detection probabilities, and integration time estimates for RV-detected exoplanets observable with Roman.

## Overview

The pipeline takes radial RV (RadVel) or RV + HGCA (Orbitize) posterior chains and forward models each posterior sample into sky-projected observables across a user-defined time window. It then estimates CGI integration times using EXOSIMS/corgietc for both optimistic and conservative noise scenarios.

## Pipeline Steps

### 1. Define Planet Parameters (`orbit_params`)

Each target is defined in the `orbit_params` dictionary with stellar mass, parallax, number of planets, and (optionally) inclination priors from HGCA/Gaia astrometry. The `posterior_type` field controls whether the chain is read as RadVel (`"radvel"`) or Orbitize (`"orbitize"`) format.

### 2. Load Posteriors (`load_posteriors`)

Reads posterior samples from CSV files (`.csv.bz2`).

- **RadVel posteriors** are in `orbit_fits/<star>/` and contain fitted RV basis parameters plus `lnprobability`.
- **Orbitize posteriors** are in `orbit_fits/Roman_RV_HGCA_Orbits/<star>/` and contain physical orbital elements (sma, ecc, inc, aop, pan, tau) plus `chi2`.

### 3. Generate the Point Cloud (`gen_point_cloud`)

Draws `nsamp` posterior samples and for each sample × each epoch in the time grid:

1. **Convert basis** — For RadVel chains: converts to synth basis -> computes Msini -> draws inclination (fixed, Gaussian from HGCA, or uniform in cos i with a critical inclination ceiling at the hydrogen burning limit) -> derives true mass, semi-major axis, and Thiele–Innes constants. For Orbitize chains: reads orbital elements directly from the posterior.
2. **Compute sky separation** — Calls `orbitize.kepler.calc_orbit` to get RA/Dec offsets in mas.
3. **Compute 3D geometry** — Solves Kepler's equation for the true anomaly, then projects via Thiele–Innes to get the z-component (line-of-sight distance) and full 3D orbital radius.
4. **Derive physical properties** — Estimates planet radius from mass using a mass–radius relation capped at 1 RJup and computes the phase angle from the z component.

Output is a pickle containing 2D arrays (n_epochs × n_samples) for separation, RA/Dec offsets, phase angle, orbital radius, planet mass, planet radius, inclination, and ln-likelihood weights.

### 4. Compute Flux Contrast & Detectability (`process_planet_band`)

For each wavelength band:

1. **Albedo** — Draws geometric albedos from a Gaussian (band-dependent mean/sigma) -> TODO: ADD GRID MODEL ALBEDOS
2. **Lambert phase function** — Computes phase angle.
3. **Flux contrast** — Computes flux contrast Fp/Fs
4. **Detection mask** — Compares flux contrast against a pre-loaded CGI contrast curve (10 hr int times) at the nearest angular separation; marks each sample as detectable or not.
5. **Detection probability** — Likelihood-weighted fraction of detectable samples at each epoch.

### 5. Estimate Integration Times (`calc_integration_time_for_cloud`)

This is the most computationally intensive step. For each epoch where the detection probability exceeds a minimum threshold (`min_det_prob`, default 0.01):

1. **Set up EXOSIMS** — Builds a `TargetList` for the host star's HIP number using the CGI noise JSON spec. Retrieves the `OpticalSystem`, which encodes throughput, detector noise, performance for **optimistic** (`OPT_*`) and **conservative** (`CON_*`) scenarios.
2. **Subsample the posterior** — Optionally draws `n_inttime_samples` (default 2000) from the full point cloud, weighted by ln-likelihood, to keep runtime manageable.
3. **Per-epoch loop** — For each epoch:
   - Pulls separation (mas), flux contrast, and orbital radius for each posterior sample.
   - Converts contrast to delta-magnitude (−2.5 log10 Fp/Fs).
   - Masks samples outside the coronagraph field of view (IWA/OWA).
   - Scales the exozodiacal dust brightness by 1/r² (local zodiacal light is held fixed).
   - Calls `OpticalSystem.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)` for both optc and con modes. This returns the exposure time (in days) required to reach the target SNR (default 5) for each sample.
   - Caps results at `max_inttime_hours` (default 1000 h); anything longer is set to a NaN.
4. **Output** — Adds `integration_time_hours_opt` and `integration_time_hours_con` arrays (n_epochs × n_samples) to the point cloud.

### 6. Observation Windows

Computes solar keepout constraints and bulge constraints using roman_pointing.

### 7. Summary Statistics (`gen_summary_csv`)

Collapses the (n_epochs × n_samples) arrays into per-epoch weighted percentiles (2.5th, 16th, 50th, 84th, 97.5th) plus weighted mean/std for all quantities: separation, orbital radius, phase angle, Lambert phase, flux contrast, and integration time. Integration-time percentiles use `weighted_percentile_nan` so that out-of-FOV NaN epochs propagate as gaps rather than biasing the statistics.

### 8. Plotting (`plot_orbital_parameters`)

Generates a multi-panel figure:

- **Left panel** (if posterior samples provided): 2D sky plane orbit tracks with IWA/OWA circles for narrow and wide FOV.
- **Right panels** (stacked time series): detection probability, angular separation, phase angle, flux contrast, and integration time (optimistic + conservative with 68%/95% credible intervals). Solar keepout and galactic bulge windows are overlaid as shaded regions.

## Default Configuration (Most of these can be changed as per notebook demo!)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_snr` | 5 | SNR threshold for integration time calc |
| `obs_mode` | `IMG_NFB1_HLC` | CGI observing mode |
| `n_inttime_samples` | 2000 | Posterior samples used for int. time (or `'all'`) |
| `max_inttime_hours` | 1000 | Cap on integration time per sample |
| `min_det_prob` | 0.01 | Skip int. time calc if detection prob. below this |
| `time_interval` | 1 day | Epoch spacing in the time grid |

## Dependencies

- `radvel`, `orbitize` — Posterior I/O and Keplerian orbit computation
- `EXOSIMS`, `corgietc` — Roman CGI noise model and integration time calculator
- `astropy`, `astroquery` — Time handling, unit conversions, SIMBAD queries
- `numpy`, `pandas`, `matplotlib` — Numerics, data wrangling, plotting

## File Structure

```
orbit_fits/
├── <star>/                          # RadVel posteriors (*.csv.bz2)
├── Roman_RV_HGCA_Orbits/<star>/     # Orbitize RV+HGCA posteriors (*.csv.bz2)
outputs/
├── outputs_radvel/                  # Point cloud pickles (RV-only)
├── outputs_orbitize/                # Point cloud pickles (RV+HGCA)
roman_table.py                       # Core library (this file)
roman_table.ipynb                    # Primary analysis notebook
detection_probabilities.ipynb        # Detection probability & int. time notebook
```
