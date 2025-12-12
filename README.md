# RV Planet Angular Separation & Phase Angle Generator

Computes sky-projected separations, 3D orbital radii, true anomalies, and phase angles
for RV-detected exoplanets using RadVel posterior samples. Includes inclination
sampling options and weighted percentiles using posterior log-likelihoods.

## Features
- Reads RadVel posteriors from `orbit_fits/<planet>/`
- Computes RA/Dec offsets, separation (mas), 3D radius (AU), phase angle, true anomaly
- Inclination modes:
    • random (uniform in cos i)
    • gaussian (from orbit_params)
    • user Gaussian (“75±10”)
    • fixed (“90”)
- Posterior weighting: uses exp(lnprob - lnprob_max)
- Outputs CSV with median/CI for all quantities
- Optional plots: 2D sky-plane orbit + time-series panels

## Usage
python roman-table.py \
    --planet HD_154345 \
    --start-date 2027-01-01 \
    --end-date 2031-06-01 \
    --time-interval 30 \
    --inclination random \
    --nsamp all \
    --plot

## Output
- CSV: <planet>_separations_<start>_to_<end>.csv
- Includes weighted percentiles for:
    separation (mas, AU), phase angle (deg), true anomaly (deg)
- Header includes:
    system parameters, planet mass & radius posteriors, inclination stats
- Optional PNG plot with:
    • 2D orbit (RA/Dec) with IWA/OWA
    • separation, radius, phase angle, true anomaly vs time
