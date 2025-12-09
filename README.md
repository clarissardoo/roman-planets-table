# RV Planet Angular Separation Generator

This tool computes sky projected angular separations for RV detected exoplanets using posterior samples from RadVel. It converts orbital posteriors into physical on sky separations at user selected epochs and writes results to a CSV file.

## Features
- Samples posteriors from RadVel fits stored in `orbit_fits/<planet>/`
- Computes RA and Dec offsets and angular separation in mas
- Uses weighted percentiles based on log likelihood
- Supports fixed or random inclinations (cos(i))
- Outputs separation in mas and AU for each epoch
- Writes a summary header and full table to CSV

## Usage

```bash
python roman-table.py 
    --planet HD_154345 
    --start-date 2027-01-01 
    --end-date 2031-06-01 
    --time-interval (days 1 - 365): 30 
    --inclination random 
    --nsamp all
