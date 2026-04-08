# Roman Coronagraph Target Detection Probability Pipeline

Generates projected separations, flux contrasts, detection probabilities, and integration time estimates for RV-detected exoplanets observable with the Roman Coronagraph Instrument (CGI).

## Overview

The pipeline takes RV (RadVel) or RV + HGCA astrometric (Orbitize) posterior chains and forward models each posterior sample into sky-projected observables across a user-defined time window. It then estimates CGI integration times using corgietc for both optimistic and conservative noise scenarios.

## Pipeline Steps

### 1. Define Planet Parameters (`orbit_params`)

Each target is defined in the `orbit_params` dictionary with stellar mass, parallax, number of planets, and (optionally) inclination priors from HGCA/Gaia astrometry. The `posterior_type` field controls whether the chain is read as RadVel or Orbitize format.

### 2. Load Posteriors (`load_posteriors`)

Reads posterior samples from CSV files.

- **RadVel posteriors** are in `orbit_fits/<star>/` and contain fitted RV basis parameters plus `lnprobability`.
- **Orbitize posteriors** are in `orbit_fits/Roman_RV_HGCA_Orbits/<star>/` and contain physical orbital elements (sma, ecc, inc, aop, pan, tau) plus `chi2`.

### 3. Generate Point Cloud (`gen_point_cloud`)

Draws `nsamp` posterior samples and for each sample × epoch:

1. **Convert basis** — For RadVel: converts to synth basis, computes Msini, draws inclination (fixed, Gaussian from HGCA, or uniform in cos i with critical inclination ceiling), derives true mass, SMA, and Thiele–Innes constants. For Orbitize/Octofitter: reads orbital elements directly.
2. **Compute sky separation** — Calls `orbitize.kepler.calc_orbit` for RA/Dec offsets in mas.
3. **Compute 3D geometry** — Solves Kepler's equation for true anomaly, projects via Thiele–Innes for z-component and 3D orbital radius.
4. **Derive physical properties** — Planet radius from mass–radius relation, phase angle from z-component.

Output: pickle with 2D arrays (n_epochs × n_samples) for separation, offsets, phase angle, orbital radius, mass, radius, inclination, and likelihood weights.

### 4. Compute Flux Contrast

Two options:

- **PICASO** (default) — Batalha+2018 atmospheric model grid with cloud sedimentation sampling (`fsed`), wavelength-dependent scattering and absorption. Returns physically motivated pphi per draw.
- **Lambert fallback** — Gaussian-drawn geometric albedo × Lambert phase function.

### 5. Estimate Integration Times (`calc_integration_time_for_cloud`)

For each epoch:

1. **EXOSIMS setup** — Builds a `TargetList` for the host star, retrieves the `OpticalSystem` with optimistic and conservative scenarios.
2. **Per-draw calculation** — Converts flux contrast to delta-magnitude, masks draws outside IWA/OWA, scales exozodi by 1/r2 with per-draw inclination correction, calls `calc_intTime` for both scenarios.
3. **Output** — `integration_time_hours_opt/con` arrays (n_epochs × n_samples), capped at 1000h.

### 6. Feasible Detection Probability (`compute_feasible_pdet`)

For each integration time budget (e.g. 10h, 100h): fraction of posterior draws with finite integration time <= budget. This is the headline metric.

### 7. Observation Windows (`compute_observation_windows`)

Solar keepout and Galactic Bulge constraints via `roman_pointing`.

### 8. Summary & Plotting

- **`gen_summary_csv`** — Per-epoch weighted percentiles (2.5th, 16th, 50th, 84th, 97.5th) for all quantities.
- **`plot_contrast_comparison`** — Multi-panel time-series: detection probability, separation, phase angle, flux contrast, integration time with observation windows overlaid.
- **`plot_inttime_histogram`** — Integration time distribution at a single epoch or peak-probability epoch in a date range.
- **`plot_orbital_parameters`** — Time-series with optional 2D sky-plane orbit panel.
  
## Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_snr` | 5 | SNR threshold for integration time |
| `int_times_hr` | [10, 100] | Integration time budgets for feasible pdet |
| `contrast_penalty` | 2.0 | Multiplier for degraded contrast scenario |
| `n_inttime_samples` | 10,000 | Posterior draws used for integration time (computationally heavy, so we downsample. 10,000 was sweet spot between computation time and consistent output) |
| `max_inttime_hours` | 1000 | Cap on integration time per draw |
| `time_interval` | 10 days | Epoch spacing |
| `nsamp` | 100,000 | Posterior draws for point cloud |

## Dependencies

- `radvel`, `orbitize` — Posterior I/O and Keplerian orbit computation
- `EXOSIMS`, `corgietc`, `cgi_noise` — Roman CGI noise model and integration time calculator
- `corgidb` — Atmospheric model photometry grid (PICASO)
- `roman_pointing` — Solar angle and pointing constraints
- `astropy`, `astroquery` — Time, units, SIMBAD queries
- `numpy`, `pandas`, `matplotlib` — Numerics, data, plotting
