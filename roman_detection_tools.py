import os
import json
import copy
import time as _time

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance, BarycentricMeanEcliptic
from astroquery.simbad import Simbad
from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions

import EXOSIMS.Prototypes.TargetList
import EXOSIMS.Prototypes.TimeKeeping
import EXOSIMS.Observatory.ObservatoryL2Halo

from roman_orbit_tools import (
    orbit_params, display_name,
    weighted_mean, weighted_percentile, weighted_percentile_nan,
    weighted_std,
)



_EXOSIMS_SPECS = None


def _get_exosims_specs():
    global _EXOSIMS_SPECS
    if _EXOSIMS_SPECS is None:
        scriptfile = os.path.join(
            os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json")
        with open(scriptfile, "r") as f:
            _EXOSIMS_SPECS = json.loads(f.read())
        _EXOSIMS_SPECS["modules"]["StarCatalog"] = "HIPfromSimbad"
    return _EXOSIMS_SPECS


WFOV_PLANETS = {'eps_Eri_b'}
ALL_BANDS = [1, 3, 4]


def get_planet_config(planet):
    """Return (bands, obs_mode_dict, contrast_curve_dict, show_wfov).

    NOTE: contrast_curve_dict values are set to None here — the caller
    must load the actual contrast curves and pass them where needed.
    """
    if planet in WFOV_PLANETS:
        bands = [1, 4]
        obs_modes = {1: 'IMG_WFB1_SPC', 4: 'IMG_WFB4_SPC'}
        show_wfov = True
    else:
        bands = [1]
        obs_modes = {
            1: 'IMG_NFB1_HLC',
            3: 'SPEC_NFB3_SPC',
            4: 'IMG_WFB4_SPC',
        }
        show_wfov = False
    return bands, obs_modes, show_wfov

def calc_integration_time_for_cloud(
    planet, point_cloud, target_snr=5, obs_mode='IMG_NFB1_HLC', band=1,
    params=None, n_inttime_samples=None, launch_date=None,
    IWA_mas=None, OWA_mas=None, max_inttime_hours=1000,
    n_zodis=1, exozodi_inc_override=None, contrast_degradation=1.0,
):
    """Compute per-draw integration times for a point cloud.

    Adds to point_cloud:
        integration_time_hours_opt/con   (n_epochs, n_samples)
        integration_time_days_opt/con
        integration_time_sample_indices
    """
    if params is None:
        params = orbit_params[planet]
    if launch_date is None:
        launch_date = Time('2026-09-28T00:00:00.000', format='isot')

    star = params['star']
    HIP_name = Simbad.query_objectids(
        star, criteria="ident.id LIKE 'HIP%'")['id'][0]
    HIP_num = int(HIP_name.split('HIP')[1])
    TL = EXOSIMS.Prototypes.TargetList.TargetList(
        **copy.deepcopy(_get_exosims_specs()), catalogpath=[HIP_num])
    OS = TL.OpticalSystem
    ZL = TL.ZodiacalLight
    sInd = 0

    mode_opt = list(filter(
        lambda m: m['Scenario'] == f'OPT_{obs_mode}', OS.observingModes))[0]
    mode_con = list(filter(
        lambda m: m['Scenario'] == f'CON_{obs_mode}', OS.observingModes))[0]
    mode_opt['SNR'] = target_snr
    mode_con['SNR'] = target_snr
    mode_opt['contrast_degradation'] = contrast_degradation
    mode_con['contrast_degradation'] = contrast_degradation

    if IWA_mas is None:
        IWA_mas = mode_opt['syst']['IWA'].to(u.mas).value
    if OWA_mas is None:
        OWA_mas = mode_opt['syst']['OWA'].to(u.mas).value

    if point_cloud['epoch_mjd'].ndim == 2:
        epoch_mjd = point_cloud['epoch_mjd'][:, 0]
    else:
        epoch_mjd = point_cloud['epoch_mjd']
    n_epochs = len(epoch_mjd)

    total_samples = point_cloud['sep_mas'].shape[1]
    lnlike = point_cloud['ln_likelihood']
    if lnlike.ndim == 2:
        lnlike = lnlike[0]
    w = np.exp(lnlike - np.max(lnlike))
    w /= w.sum()

    if n_inttime_samples is None or n_inttime_samples == 'all':
        sample_indices = np.arange(total_samples)
    else:
        n_inttime_samples = min(int(n_inttime_samples), total_samples)
        sample_indices = np.random.choice(
            total_samples, size=n_inttime_samples, replace=False, p=w)
    n_samples = len(sample_indices)

    # Exozodi inclination correction
    inc_all = point_cloud['inc_deg']
    if inc_all.ndim == 2:
        inc_all = inc_all[0]

    if exozodi_inc_override is not None:
        fbeta_all = np.full(
            n_samples,
            float(ZL.calc_fbeta(np.array([exozodi_inc_override]) * u.deg)[0]))
    else:
        inc_sampled = inc_all[sample_indices]
        fbeta_all = ZL.calc_fbeta(inc_sampled * u.deg)

    Tint_opt_hours = np.full((n_epochs, n_samples), np.nan)
    Tint_con_hours = np.full((n_epochs, n_samples), np.nan)

    print(f'Calculating integration times for {planet}, band {band}')
    print(f'  {n_epochs} epochs x {n_samples} samples')
    print(f'  n_zodis={n_zodis}, '
          f'fbeta range=[{fbeta_all.min():.3f}, {fbeta_all.max():.3f}]')
    print(f'  IWA={IWA_mas:.1f} mas, OWA={OWA_mas:.1f} mas')
    print(f'  Max integration time: {max_inttime_hours} hours')
    print()

    scenario_start = _time.time()
    epochs_computed = 0
    for i_epoch in range(n_epochs):
        elapsed = _time.time() - scenario_start
        rate = epochs_computed / elapsed if elapsed > 0 else 0
        remaining = (n_epochs - i_epoch) / rate if rate > 0 else 0
        epoch_year = Time(epoch_mjd[i_epoch], format='mjd').decimalyear
        print(f'  Epoch {i_epoch + 1:3d}/{n_epochs} '
              f'(year={epoch_year:.3f}) | '
              f'{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining | '
              f'{epochs_computed} computed, '
              f'{i_epoch - epochs_computed} skipped', end='\r')

        seps = point_cloud['sep_mas'][i_epoch, sample_indices]
        contrasts_raw = point_cloud['flux_contrast'][i_epoch, sample_indices]
        orb_rad = point_cloud['orbital_radius_au'][i_epoch, sample_indices]

        valid_contrast = contrasts_raw > 0
        contrasts_dmag = np.where(
            valid_contrast, -2.5 * np.log10(contrasts_raw), np.nan)

        valid = np.isfinite(contrasts_dmag)
        valid &= (seps >= IWA_mas) & (seps <= OWA_mas)
        if not np.any(valid):
            continue

        seps_v = seps[valid]
        contrast_v = contrasts_dmag[valid]
        orb_rad_v = orb_rad[valid]
        n_valid = int(np.sum(valid))

        fbeta_v = fbeta_all[valid]
        JEZ_v = (TL.JEZ0[mode_opt["hex"]][sInd]
                 * n_zodis * fbeta_v / orb_rad_v ** 2)
        fZ_v = np.repeat(ZL.fZ0, n_valid)
        sInds_v = np.zeros(n_valid, dtype=int)

        for mode, arr in [(mode_opt, Tint_opt_hours),
                          (mode_con, Tint_con_hours)]:
            try:
                Tint = OS.calc_intTime(
                    TL, sInds_v, fZ_v, JEZ_v,
                    contrast_v, seps_v * u.mas, mode)
                t_hours = Tint.to_value(u.h)
                t_hours[t_hours > max_inttime_hours] = max_inttime_hours
                arr[i_epoch, valid] = t_hours
            except Exception as e:
                print(f'\n  Warning: calc_intTime failed at epoch '
                      f'{i_epoch}: {e}')
                continue

        epochs_computed += 1

    print(f'\nDone. Total time: '
          f'{(_time.time() - scenario_start) / 60:.1f} minutes')
    print(f'  {epochs_computed}/{n_epochs} epochs computed, '
          f'{n_epochs - epochs_computed} skipped')

    for label, arr in [('opt', Tint_opt_hours), ('con', Tint_con_hours)]:
        valid_vals = arr[np.isfinite(arr)]
        n_nan = np.sum(np.isnan(arr))
        stats = (f', range {valid_vals.min():.2f}-{valid_vals.max():.2f} h '
                 f'(median {np.median(valid_vals):.2f} h)'
                 if len(valid_vals) > 0 else '')
        print(f'  {label}: {len(valid_vals)} valid, {n_nan} NaN{stats}')

    point_cloud['integration_time_hours_opt'] = Tint_opt_hours
    point_cloud['integration_time_hours_con'] = Tint_con_hours
    point_cloud['integration_time_days_opt'] = Tint_opt_hours / 24.0
    point_cloud['integration_time_days_con'] = Tint_con_hours / 24.0
    point_cloud['integration_time_sample_indices'] = sample_indices

    return point_cloud

def calculate_integration_time(planet,target_snr,band=1,obs_mode='IMG_NFB1_HLC'):
    # this is Arthur Vigan's function which does int time calculation using percentiles.
    # THe cloud function is what we use in output plots, see above
    #in notebook
    target=planet
    SNR=target_snr
    obs_mode=obs_mode
    band=band

    data=load_point_cloud(target)

    # star Hipparcos identifier
    star=target[:-2]
    HIP_name=Simbad.query_objectids(star,criteria="ident.id LIKE 'HIP%'")['id'][0]
    HIP_num=int(HIP_name.split('HIP')[1])

    # generate the target list
    TL=EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs),catalogpath=[HIP_num])
    OS=TL.OpticalSystem
    ZL=TL.ZodiacalLight

    Obs=EXOSIMS.Observatory.ObservatoryL2Halo.ObservatoryL2Halo()
    sInd=0

    TK=EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    launch_date=Time('2026-09-28T00:00:00.000',format='isot')

    for scenario in ('opt','con'):
        print(f'Scenario = {scenario}')

        mode_name=f'{scenario.upper()}_{obs_mode}'
        mode=list(filter(lambda mode:mode['Scenario']==mode_name,OS.observingModes))[0]

        mode['SNR']=target_snr

        fZ=np.repeat(TL.ZodiacalLight.fZ0,1)

        for irow,row in data.iterrows():
            mjd=row['mjd']
            dyear=row['decimal_year']
            sep_med=row['separation_mas_median']
            orb_rad_au=row['orbital_radius_au_median']
            contrast_med=-2.5*np.log10(row['flux_contrast_median'])

            print(f'> {dyear:.2f}, sep={sep_med:6.1f} mas, contrast={contrast_med:.1f} mag',end='\r')

            TK.allocate_time((mjd-launch_date.mjd)*u.d)

            JEZ=np.repeat(TL.JEZ0[mode["hex"]][sInd]/orb_rad_au**2,1)

            Tint=OS.calc_intTime(TL,[sInd],fZ,JEZ,np.array([contrast_med]),np.array([sep_med])*u.mas,mode)
            data.loc[irow,f'integration_time_days_{scenario}']=Tint.to_value(u.d)
            data.loc[irow,f'integration_time_hours_{scenario}']=Tint.to_value(u.h)

    return data['integration_time_hours_opt'].values,data['integration_time_hours_con'].values


def get_iwa_owa(planet, obs_mode, target_snr=5):
    """Get IWA/OWA in mas for a given planet and observing mode."""
    params = orbit_params[planet]
    star = params['star']
    HIP_name = Simbad.query_objectids(
        star, criteria="ident.id LIKE 'HIP%'")['id'][0]
    HIP_num = int(HIP_name.split('HIP')[1])

    TL = EXOSIMS.Prototypes.TargetList.TargetList(
        **copy.deepcopy(_get_exosims_specs()), catalogpath=[HIP_num])
    OS = TL.OpticalSystem

    mode = list(filter(
        lambda m: m['Scenario'] == f'OPT_{obs_mode}', OS.observingModes))[0]

    IWA_mas = mode['syst']['IWA'].to(u.mas).value
    OWA_mas = mode['syst']['OWA'].to(u.mas).value
    return IWA_mas, OWA_mas
# =====================================================================
# Feasible detection probability
# =====================================================================

def compute_feasible_pdet(point_cloud, weights, max_hours_list=None):
    """Compute feasible detection probability from integration time arrays.

    Parameters
    ----------
    point_cloud : dict
        Must contain integration_time_hours_opt/con and
        integration_time_sample_indices.
    weights : (n_total_samples,) array
        Likelihood weights for ALL posterior samples.
    max_hours_list : list of float
        Integration time budgets to evaluate. Default [100].

    Returns
    -------
    scenarios : dict of label -> (n_epochs,) pdet array
    """
    if max_hours_list is None:
        max_hours_list = [100]

    inttime_indices = point_cloud['integration_time_sample_indices']
    inttime_w = weights[inttime_indices]
    inttime_w /= inttime_w.sum()

    scenarios = {}
    for t_hr in max_hours_list:
        for scenario, key in [('opt', 'integration_time_hours_opt'),
                              ('con', 'integration_time_hours_con')]:
            arr = point_cloud[key]
            is_feasible = np.isfinite(arr) & (arr <= t_hr)
            pdet = np.average(
                is_feasible.astype(float), axis=1)
            label = f'{t_hr}h {scenario}'
            scenarios[label] = pdet
            print(f'  {label}: pdet [{pdet.min():.3f}, {pdet.max():.3f}]')

    return scenarios


# =====================================================================
# Contrast curve generation
# =====================================================================

def gen_corgietc_contrast_curves(
    planet, obs_mode, int_times_hr=None, target_snr=5, n_wa=50,
    repr_orb_rad_au=3.0, n_zodis=1, exozodi_inc_deg=None,
):
    """Generate corgietc contrast curves at fixed integration times.

    Returns
    -------
    contrast_curves_by_tint : dict
        Keyed by (t_hr, scenario) -> (sep_mas_array, contrast_array)
    IWA_mas, OWA_mas : float
    TL, OS : EXOSIMS objects (reusable)
    """
    if int_times_hr is None:
        int_times_hr = [10, 100]

    params = orbit_params[planet]
    star = params['star']
    HIP_name = Simbad.query_objectids(
        star, criteria="ident.id LIKE 'HIP%'")['id'][0]
    HIP_num = int(HIP_name.split('HIP')[1])

    TL = EXOSIMS.Prototypes.TargetList.TargetList(
        **copy.deepcopy(_get_exosims_specs()), catalogpath=[HIP_num])
    OS = TL.OpticalSystem
    sInd = 0

    mode_opt = list(filter(
        lambda m: m['Scenario'] == f'OPT_{obs_mode}', OS.observingModes))[0]
    mode_con = list(filter(
        lambda m: m['Scenario'] == f'CON_{obs_mode}', OS.observingModes))[0]
    mode_opt['SNR'] = target_snr
    mode_con['SNR'] = target_snr

    IWA_as = mode_opt['syst']['IWA'].to(u.arcsec).value
    OWA_as = mode_opt['syst']['OWA'].to(u.arcsec).value
    IWA_mas = IWA_as * u.arcsec.to(u.mas)
    OWA_mas = OWA_as * u.arcsec.to(u.mas)

    WA_as = np.linspace(IWA_as * 1.01, OWA_as * 0.99, n_wa)
    WA_q = WA_as * u.arcsec

    ZL = TL.ZodiacalLight
    fZ = np.repeat(ZL.fZ0, n_wa)
    if exozodi_inc_deg is not None:
        fbeta = float(
            ZL.calc_fbeta(np.array([exozodi_inc_deg]) * u.deg)[0])
    else:
        fbeta = 1.0
    JEZ = np.repeat(
        TL.JEZ0[mode_opt["hex"]][sInd] * n_zodis * fbeta
        / repr_orb_rad_au ** 2, n_wa)
    sInds = np.zeros(n_wa, dtype=int)

    contrast_curves_by_tint = {}
    for t_hr in int_times_hr:
        intTimes = np.full(n_wa, t_hr / 24.0) * u.d
        for scenario, mode, label in [('opt', mode_opt, 'optimistic'),
                                       ('con', mode_con, 'conservative')]:
            print(f'    t_int={t_hr}h, {label}...')
            try:
                dMag = OS.calc_dMag_per_intTime(
                    intTimes, TL, sInds, fZ, JEZ, WA_q, mode)
                fc = 10 ** (-dMag / 2.5)
                sep_mas = WA_as * u.arcsec.to(u.mas)
                contrast_curves_by_tint[(t_hr, scenario)] = (
                    sep_mas.copy(), fc.copy())
                print(f'      contrast: {fc.min():.2e} – {fc.max():.2e}')
            except Exception as e:
                print(f'      WARNING: calc_dMag_per_intTime failed: {e}')
                contrast_curves_by_tint[(t_hr, scenario)] = None

    return contrast_curves_by_tint, IWA_mas, OWA_mas, TL, OS



def get_GB_sunang(ts):
    """Sun angle to Galactic Bulge reference (Sgr A*) at times ts."""
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
    res = simbad.query_object("Sagittarius A*")
    gb = SkyCoord(
        res["ra"].value.data[0], res["dec"].value.data[0],
        unit=(res["ra"].unit, res["dec"].unit),
        frame="icrs",
        distance=Distance(8 * u.kpc),
        pm_ra_cosdec=0 * res["pmra"].unit,
        pm_dec=0 * res["pmdec"].unit,
        radial_velocity=0 * res["rvz_radvel"].unit,
        equinox="J2000", obstime="J2000",
    ).transform_to(BarycentricMeanEcliptic)
    sun_ang, _, _, _ = calcRomanAngles(gb, ts, getL2Positions(ts))
    return sun_ang


def get_targ_sunang(star, ts):
    """Sun angle to a target star at times ts."""
    simbad_name = " ".join(star.split("_"))
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
    res = simbad.query_object(simbad_name)
    target = SkyCoord(
        res["ra"].value.data[0], res["dec"].value.data[0],
        unit=(res["ra"].unit, res["dec"].unit),
        frame="icrs",
        distance=Distance(
            parallax=res["plx_value"].value.data[0] * res["plx_value"].unit),
        pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
        pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
        radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
        equinox="J2000", obstime="J2000",
    ).transform_to(BarycentricMeanEcliptic)
    sun_ang, _, _, _ = calcRomanAngles(target, ts, getL2Positions(ts))
    return sun_ang


def compute_observation_windows(point_cloud, planet):
    """Add GB_not_observable and targ_observable to point_cloud."""
    if point_cloud['epoch_mjd'].ndim == 2:
        epoch_mjd = point_cloud['epoch_mjd'][:, 0]
    else:
        epoch_mjd = point_cloud['epoch_mjd']

    times = Time(epoch_mjd, format='mjd')

    sun_ang_ref = get_GB_sunang(times)
    point_cloud['GB_not_observable'] = ~(
        (sun_ang_ref.to_value(u.deg) > 54)
        & (sun_ang_ref.to_value(u.deg) < 126))

    star = orbit_params[planet]['star']
    sun_ang_targ = get_targ_sunang(star, times)
    point_cloud['targ_observable'] = (
        (sun_ang_targ.to_value(u.deg) > 54)
        & (sun_ang_targ.to_value(u.deg) < 126))

    return point_cloud