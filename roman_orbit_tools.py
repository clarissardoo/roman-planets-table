import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau, tau_to_tp
from orbitize.kepler import calc_orbit


# =====================================================================
# Planet parameters
# =====================================================================

orbit_params = {
    "ups_And_d": {
        'star': 'ups_And', 'pl_letter': 'd',
        "basis": "per tc secosw sesinw k",
        "m0": 1.29419667430000, "m0_err": 0.04122482369025,
        "plx": 74.1940, "plx_err": 0.2083,
        "n_planets": 3, "pl_num": 3, "g_mag": 3.966133,
        "inc_mean": 23.758, "inc_sig": 1.316,
        "posterior_type": "radvel",
    },
    "eps_Eri_b": {
        'star': 'eps_Eri', 'pl_letter': 'b',
        "basis": "per tc secosw sesinw k",
        "m0": 0.82, "m0_err": 0.02,
        "plx": 310.5773, "plx_err": 0.1355,
        "n_planets": 1, "pl_num": 1, "g_mag": 3.465752,
        "inc_mean": 78.810, "inc_sig": 29.340,
        "posterior_type": "radvel",
    },
    "14_Her_b": {
        'star': '14_Her', 'pl_letter': 'b',
        "m0": 0.98, "m0_err": 0.04,
        "plx": 55.8657, "plx_err": 0.0291,
        "n_planets": 2, "pl_num": 1, "g_mag": 6.3830000,
        "posterior_type": "orbitize",
    },
    "47_UMa_c": {
        "star": "47_UMa", 'pl_letter': 'c',
        "basis": "per tc secosw sesinw k",
        "m0": 1.0051917028549999, "m0_err": 0.0468882076437500,
        "plx": 72.0070, "plx_err": 0.0974,
        "n_planets": 3, "pl_num": 2, "g_mag": 4.866588,
        "posterior_type": "radvel",
    },
    "HD_154345_b": {
        'star': 'HD_154345', 'pl_letter': 'b',
        "basis": "per tc secosw sesinw k",
        "m0": 0.88, "m0_err": 0.09,
        "plx": 54.7359, "plx_err": 0.0176,
        "n_planets": 1, "pl_num": 1, "g_mag": 6.583667,
        "inc_mean": 69, "inc_sig": 13,
        "posterior_type": "radvel",
    },
    "HD_190360_b": {
        'star': 'HD_190360', 'pl_letter': 'b',
        "basis": "per tc secosw sesinw k",
        "m0": 1.0, "m0_err": 0.1,
        "plx": 62.4865, "plx_err": 0.0354,
        "n_planets": 2, "pl_num": 1, "g_mag": 5.552787,
        "inc_mean": 80.2, "inc_sig": 23.2,
        "posterior_type": "radvel",
    },
    "HD_217107_c": {
        'star': 'HD_217107', 'pl_letter': 'c',
        "basis": "per tc secosw sesinw k",
        "m0": 1.05963082882500, "m0_err": 0.04470613802572,
        "plx": 49.7846, "plx_err": 0.0263,
        "n_planets": 2, "pl_num": 2, "g_mag": 5.996743,
        "inc_mean": 89.3, "inc_sig": 9.0,
        "posterior_type": "radvel",
    },
    "HD_114783_c": {
        'star': 'HD_114783', 'pl_letter': 'c',
        "basis": "per tc secosw sesinw k",
        "m0": 0.90, "m0_err": 0.04,
        "plx": 47.5529, "plx_err": 0.0291,
        "n_planets": 2, "pl_num": 2, "g_mag": 7.330857,
        "inc_mean": 159, "inc_sig": 6,
        "posterior_type": "radvel",
    },
}


def display_name(planet_name):
    """'eps_Eri_b' -> 'Eps Eri b'"""
    replaced = " ".join(planet_name.split("_"))
    return replaced[0].upper() + replaced[1:]


def parse_inclination(inc_str):
    """Parse inclination string from CLI/notebook input.

    Returns (mode, value, sigma) tuple:
        ('random', None, None)
        ('gaussian', None, None)          — use inc_mean/inc_sig from params
        ('user_gaussian', mean, sigma)
        ('fixed', value, None)
    """
    inc_str = inc_str.strip().lower()
    if inc_str == 'random':
        return ('random', None, None)
    elif inc_str == 'gaussian':
        return ('gaussian', None, None)

    import re
    for pattern in [r'^(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)$',
                    r'^(\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)$']:
        match = re.match(pattern, inc_str)
        if match:
            mean_val = float(match.group(1))
            sigma_val = float(match.group(2))
            if mean_val < 0 or mean_val > 180:
                raise ValueError(f"Inclination mean must be 0–180 deg (got {mean_val})")
            if sigma_val <= 0:
                raise ValueError(f"Inclination sigma must be > 0 (got {sigma_val})")
            return ('user_gaussian', mean_val, sigma_val)

    try:
        value = float(inc_str)
        if value < 0 or value > 180:
            raise ValueError(f"Inclination must be 0–180 deg (got {value})")
        return ('fixed', value, None)
    except ValueError as e:
        if "Inclination must be" in str(e):
            raise
        raise ValueError(f"Invalid inclination format: '{inc_str}'")



def weighted_percentile(data, weights, percentile):
    """Weighted percentile along axis=1 of a 2-D array."""
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        idx = np.argsort(data[i, :])
        sd = data[i, idx]
        sw = weights[idx]
        cs = np.cumsum(sw)
        j = np.searchsorted(cs, percentile / 100.0)
        if j >= len(sd):
            j = len(sd) - 1
        result[i] = sd[j]
    return result


def weighted_percentile_nan(data, weights, percentile):
    """Like weighted_percentile but skips NaN values."""
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        valid = ~np.isnan(data[i, :])
        if np.sum(valid) < 20:
            result[i] = np.nan
            continue
        d = data[i, valid]
        w = weights[valid]
        w = w / w.sum()
        idx = np.argsort(d)
        sd = d[idx]
        sw = w[idx]
        cs = np.cumsum(sw)
        j = np.searchsorted(cs, percentile / 100.0)
        if j >= len(sd):
            j = len(sd) - 1
        result[i] = sd[j]
    return result


def weighted_mean(data, weights):
    if data.ndim == 1:
        return np.average(data, weights=weights)
    return np.average(data, axis=1, weights=weights)


def weighted_std(data, weights):
    if data.ndim == 1:
        mean = np.average(data, weights=weights)
        return np.sqrt(np.average((data - mean) ** 2, weights=weights))
    mean = np.average(data, axis=1, weights=weights)
    return np.sqrt(np.average((data - mean[:, np.newaxis]) ** 2,
                              axis=1, weights=weights))


# =====================================================================
# Orbit computation
# =====================================================================

def compute_sep(
        df, epochs, basis=None, m0=None, m0_err=None, plx=None, plx_err=None,
        n_planets=1, pl_num=1, override_inc=None, override_lan=None,
        inc_mean=None, inc_sig=None, user_inc_mean=None, user_inc_sig=None,
        posterior_type=None):
    """
    Computes a sky-projected angular separation posterior given either a
    RadVel or Orbitize posterior DataFrame.

    Args:
        df (pd.DataFrame): Posterior samples (RadVel or Orbitize format)
        epochs (np.array of astropy.time.Time): epochs at which to compute separations
        basis (str): basis string for RadVel posteriors. Not used for Orbitize.
        m0 (float): median of primary mass (Gaussian). Required for RadVel, optional for Orbitize.
        m0_err (float): 1sigma error of primary mass. For RadVel posteriors.
        plx (float): median of parallax (Gaussian). Required for RadVel, optional for Orbitize.
        plx_err: 1sigma error of parallax. For RadVel posteriors.
        n_planets (int): total number of planets in posterior
        pl_num (int): planet number (e.g. 'per1' or 'sma1' implies pl_num == 1)
        override_inc (float or str): Fixed inclination (deg) for RadVel only
        override_lan (float): Fixed longitude of ascending node (deg) for RadVel only
        inc_mean (float): Mean inclination (deg) for RadVel Gaussian sampling
        inc_sig (float): Std dev inclination (deg) for RadVel Gaussian sampling
        user_inc_mean (float): User-provided mean inclination (deg) for RadVel
        user_inc_sig (float): User-provided std dev inclination (deg) for RadVel
        posterior_type (str): 'radvel' or 'orbitize' (None defaults to 'radvel')

    Returns:
        tuple of:
            seps (np.array): sky-projected angular separations [mas] (n_epochs x n_samples)
            raoff (np.array): RA offsets [mas]
            deoff (np.array): Dec offsets [mas]
            m_pl (np.array): planet masses [M_sun]
            inc (np.array): inclinations [radians]
            true_anomaly (np.array): true anomaly [radians]
            z_au (np.array): z component [AU]
            r_au (np.array): 3D orbital radius [AU]
            parallax (np.array): parallax [mas]
    """

    chain_len = len(df)
    tau_ref_epoch = 58849

    if posterior_type is None:
        posterior_type = 'radvel'

    if posterior_type == 'orbitize':
        print("Using Orbitize posterior format...")
        sma = df[f'sma{pl_num}'].values
        ecc = df[f'ecc{pl_num}'].values
        inc = np.radians(df[f'inc{pl_num}'].values) * 180 / np.pi
        omega_pl_rad = np.radians(df[f'aop{pl_num}'].values) * 180 / np.pi
        lan = np.radians(df[f'pan{pl_num}'].values) * 180 / np.pi
        tau = df[f'tau{pl_num}'].values

        if 'm0' in df.columns:
            m_st = df['m0'].values
        elif m0 is not None:
            print(f"Warning: Stellar mass (m0) not in posterior, using m0={m0}")
            m_st = np.full(chain_len, m0)
        else:
            raise ValueError("Need stellar mass (m0) in posterior or m0 parameter")

        planet_mass_col = f'm{pl_num}'
        if planet_mass_col in df.columns:
            m_pl = df[planet_mass_col].values
        else:
            print(f"Warning: Planet mass ({planet_mass_col}) not found, using fallback")
            m_pl = None

        if m_pl is None:
            m_pl = 0.001 * m_st
            print("Warning: Using placeholder planet mass estimate")

        mtot = m_st + m_pl

        if 'plx' in df.columns:
            parallax = df['plx'].values
        elif 'parallax' in df.columns:
            parallax = df['parallax'].values
        elif plx is not None:
            parallax = np.random.normal(
                plx, plx_err if plx_err is not None else 0.01 * plx,
                size=chain_len)
        else:
            raise ValueError("Need parallax in posterior or plx parameter")

    else:  # radvel
        print("Using RadVel posterior format...")
        if basis is None:
            raise ValueError("basis parameter required for RadVel posteriors")
        if m0 is None:
            raise ValueError("m0 parameter required for RadVel posteriors")
        if plx is None:
            raise ValueError("plx parameter required for RadVel posteriors")

        myBasis = Basis(basis, n_planets)
        df = myBasis.to_synth(df)

        m_st = np.random.normal(m0, m0_err, size=chain_len)
        semiamp = df[f'k{pl_num}'].values
        per_day = df[f'per{pl_num}'].values
        period_yr = per_day / 365.25
        ecc = df[f'e{pl_num}'].values
        msini = (Msini(semiamp, per_day, m_st, ecc, Msini_units='Earth')
                 * (u.M_earth / u.M_sun).to(''))

        median_msini = np.median(msini)

        if user_inc_mean is not None and user_inc_sig is not None:
            inc_deg_samples = np.clip(
                np.random.normal(user_inc_mean, user_inc_sig, size=chain_len),
                0, 180)
            inc = np.radians(inc_deg_samples)
        elif override_inc is not None and override_inc != "gaussian":
            inc = np.full(chain_len, np.radians(override_inc))
        elif override_inc == "gaussian" and inc_mean is not None and inc_sig is not None:
            inc_deg_samples = np.clip(
                np.random.normal(inc_mean, inc_sig, size=chain_len), 0, 180)
            inc = np.radians(inc_deg_samples)
        else:
            crit_incrad = np.arcsin(median_msini / 0.08)
            cosi = (2.0 * np.random.random(size=chain_len)
                    * np.cos(crit_incrad)) - 1.0
            inc = np.arccos(cosi)

        m_pl = msini / np.sin(inc)
        mtot = m_st + m_pl
        sma = (period_yr ** 2 * mtot) ** (1 / 3)
        omega_pl_rad = df[f'w{pl_num}'].values + np.pi
        parallax = np.random.normal(plx, plx_err, size=chain_len)

        if override_lan is not None:
            lan = np.full(chain_len, np.radians(override_lan))
        else:
            lan = np.random.random_sample(size=chain_len) * 2.0 * np.pi

        tp_mjd = df[f'tp{pl_num}'].values - 2400000.5
        tau = tp_to_tau(tp_mjd, tau_ref_epoch, period_yr)


    raoff, deoff, vz = calc_orbit(
        epochs.mjd, sma, ecc, inc,
        omega_pl_rad, lan, tau,
        parallax, mtot, tau_ref_epoch=tau_ref_epoch)
    seps = np.sqrt(raoff ** 2 + deoff ** 2)

    n_epochs = len(epochs)
    true_anomaly = np.zeros((n_epochs, chain_len))
    x_mas = np.zeros((n_epochs, chain_len))
    y_mas = np.zeros((n_epochs, chain_len))
    z_mas = np.zeros((n_epochs, chain_len))

    # Thiele-Innes constants
    A = sma * (np.cos(omega_pl_rad) * np.cos(lan)
               - np.sin(omega_pl_rad) * np.sin(lan) * np.cos(inc))
    B = sma * (np.cos(omega_pl_rad) * np.sin(lan)
               + np.sin(omega_pl_rad) * np.cos(lan) * np.cos(inc))
    F = sma * (-np.sin(omega_pl_rad) * np.cos(lan)
               - np.cos(omega_pl_rad) * np.sin(lan) * np.cos(inc))
    G = sma * (-np.sin(omega_pl_rad) * np.sin(lan)
               + np.cos(omega_pl_rad) * np.cos(lan) * np.cos(inc))
    C = sma * np.sin(omega_pl_rad) * np.sin(inc)
    H = sma * np.cos(omega_pl_rad) * np.sin(inc)

    period_yr = (sma ** 3 / mtot) ** 0.5
    per_day = period_yr * 365.25
    tp_mjd = tau_to_tp(tau, tau_ref_epoch, period_yr)

    for i in range(n_epochs):
        n_motion = 2 * np.pi / per_day
        M = n_motion * (epochs.mjd[i] - tp_mjd)
        EA = M + ecc * np.sin(M) + ecc ** 2 * np.sin(2 * M) / 2
        for _ in range(20):
            err = EA - ecc * np.sin(EA) - M
            if np.all(np.abs(err) < 1e-15):
                break
            EA -= err / (1 - ecc * np.cos(EA))

        f = 2 * np.arctan2(
            np.sqrt(1 + ecc) * np.sin(EA / 2),
            np.sqrt(1 - ecc) * np.cos(EA / 2))
        true_anomaly[i, :] = f

        X = np.cos(EA) - ecc
        Y = np.sqrt(1 - ecc ** 2) * np.sin(EA)
        x_mas[i, :] = (B * X + G * Y) * parallax
        y_mas[i, :] = (A * X + F * Y) * parallax
        z_mas[i, :] = (C * X + H * Y) * parallax

    r_au = np.sqrt(x_mas ** 2 + y_mas ** 2 + z_mas ** 2) / parallax
    z_au = z_mas / parallax

    return seps, raoff, deoff, m_pl, inc, true_anomaly, z_au, r_au, parallax


# =====================================================================
# Posterior I/O
# =====================================================================
def load_octofitter_fits(fpath, planet_letters=None, tau_ref_epoch=58849):
    """Load an Octofitter/Pigeons FITS posterior into a DataFrame
    compatible with compute_sep (orbitize format).

    Column mapping:
        {letter}_a           -> sma{N}   (AU)
        {letter}_e           -> ecc{N}
        {letter}_i           -> inc{N}   (radians)
        ${letter}_{\\omega}$  -> aop{N}   (radians)
        ${letter}_{\\Omega}$  -> pan{N}   (radians)
        {letter}_tp          -> tau{N}   (recomputed to tau_ref_epoch)
        {letter}_mass        -> m{N}     (M_Jup -> M_sun)
        M_pri                -> m0       (M_sun)
        plx                  -> plx      (mas)
        logpost              -> chi2     (= -2 * logpost, for weighting)
    """
    from astropy.table import Table

    t=Table.read(fpath,hdu=1)
    # FITS is big-endian; convert to native byte order for pandas
    for col in t.colnames:
        if t[col].dtype.byteorder=='>':
            t[col]=t[col].astype(t[col].dtype.newbyteorder('='))
    if planet_letters is None:
        planet_letters = sorted(set(
            c.split('_')[0] for c in t.colnames
            if len(c) >= 3 and c[1] == '_' and c[0].isalpha()
            and c.split('_')[1] == 'a'
        ))

    df = pd.DataFrame()

    # System-level
    df['m0'] = t['M_pri'].data.flatten()
    df['plx'] = t['plx'].data.flatten()

    if 'logpost' in t.colnames:
        df['chi2'] = -2 * t['logpost'].data.flatten()
    elif 'loglike' in t.colnames:
        df['chi2'] = -2 * t['loglike'].data.flatten()

    # Per-planet
    Mjup_to_Msun = (u.M_jup / u.M_sun).to('')
    for n, letter in enumerate(planet_letters, start=1):
        df[f'sma{n}'] = t[f'{letter}_a'].data.flatten()
        df[f'ecc{n}'] = t[f'{letter}_e'].data.flatten()
        df[f'inc{n}'] = t[f'{letter}_i'].data.flatten()

        omega_col = f'${letter}_{{\\omega}}$'
        Omega_col = f'${letter}_{{\\Omega}}$'
        df[f'aop{n}'] = t[omega_col].data.flatten()
        df[f'pan{n}'] = t[Omega_col].data.flatten()

        df[f'm{n}'] = t[f'{letter}_mass'].data.flatten() * Mjup_to_Msun

        period_yr=t[f'{letter}_P'].data.flatten()  # years
        tp_mjd=t[f'{letter}_tp'].data.flatten()  # MJD
        df[f'tau{n}']=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)
    print(f'Loaded {len(t)} Octofitter samples, '
          f'{len(planet_letters)} planets: {planet_letters}')
    for n, letter in enumerate(planet_letters, start=1):
        print(f'  Planet {letter} (pl_num={n}): '
              f'sma={df[f"sma{n}"].median():.3f} AU, '
              f'ecc={df[f"ecc{n}"].median():.3f}, '
              f'inc={np.degrees(df[f"inc{n}"].median()):.1f} deg, '
              f'mass={df[f"m{n}"].median()/Mjup_to_Msun:.2f} M_Jup')

    return df
def load_posteriors(planet, params=None, posterior_dir='orbit_fits', format="radvel"):
    """
    Load posterior samples from either RadVel or Orbitize format.

    Args:
        planet (str): Planet name (e.g., '47_UMa_b')
        params (dict): Planet parameters dictionary
        posterior_dir (str): Base directory for posteriors
        format (str): 'radvel' or 'orbitize'

    Returns:
        pd.DataFrame: Posterior samples
    """
    if params is None:
        params = orbit_params[planet]
    star = params['star']

    if format == 'radvel':
        planet_dir = os.path.join(posterior_dir, star)
        files = list(glob.glob(os.path.join(planet_dir, "*.csv.bz2")))
        if len(files) == 0:
            raise UserWarning(f"No posterior data found for {planet} in {planet_dir}")
        if len(files) > 1:
            raise UserWarning(f"Multiple posterior files found for {planet} in {planet_dir}")
        print(f"Loading RadVel posterior from {files[0]}...")
        df = pd.read_csv(files[0])
    elif format == 'orbitize':
        planet_dir = os.path.join(posterior_dir, 'Roman_RV_HGCA_Orbits', star)
        files = list(glob.glob(os.path.join(planet_dir, "*.csv")))
        files += list(glob.glob(os.path.join(planet_dir, "*.csv.bz2")))
        if len(files) == 0:
            raise UserWarning(f"No posterior data found for {planet} in {planet_dir}")
        if len(files) > 1:
            pl_letter = params.get('pl_letter', 'b')
            matching = [f for f in files if pl_letter in os.path.basename(f).lower()]
            if len(matching) == 1:
                files = matching
            else:
                raise UserWarning(f"Multiple files found for {planet}: {files}")
        print(f"Loading Orbitize posterior from {files[0]}...")
        df = pd.read_csv(files[0])
    else:
        raise UserWarning(f"Unknown posterior format: {format}")
    return df


def get_likelihood_weights(df, posterior_type='radvel'):
    """
    Extract likelihood weights from posterior DataFrame.

    Args:
        df (pd.DataFrame): Posterior samples
        posterior_type (str): 'radvel' or 'orbitize'

    Returns:
        np.array: Normalized weights for each sample
    """

    if posterior_type == 'orbitize':
        if 'chi2' in df.columns:
            log_like = -df['chi2'].values / 2
        else:
            return np.ones(len(df)) / len(df)
    else:
        if 'lnprobability' in df.columns:
            log_like = df['lnprobability'].values
        else:
            return np.ones(len(df)) / len(df)
    weights = np.exp(log_like - np.max(log_like))
    return weights / weights.sum()


def load_point_cloud(planet, i_dir='.', start_date='2027-01-01',
                     end_date='2027-06-30', fname=None, broadcast_arrays=True):
    """Load a saved point cloud pickle."""
    if fname is None:
        planet_name = planet.replace("_", "")
        fname = f"{planet_name}_{start_date}_to_{end_date}_PointCloud.pkl"
    fpath = os.path.join(i_dir, fname)
    print(f'Loading point cloud from {fpath}')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Point cloud not found: {fpath}")
    with open(fpath, 'rb') as f:
        point_cloud = pickle.load(f)

    if broadcast_arrays:
        arr_shape = point_cloud['sep_mas'].shape
        for param, arr in point_cloud.items():
            if arr.shape == (arr_shape[0],):
                point_cloud[param] = np.full(arr_shape, arr[:, np.newaxis])
            elif arr.shape == (arr_shape[1],):
                point_cloud[param] = np.full(arr_shape, arr)
            elif arr.shape != arr_shape:
                raise ValueError(
                    f'param {param} has unexpected shape {arr.shape}, '
                    f'sep_mas has {arr_shape}.')
    return point_cloud


# =====================================================================
# Point cloud generation
# =====================================================================

def gen_point_cloud(planet, post_df, params=None, output_dir='.',
                    start_date='2027-01-01', end_date='2027-06-30',
                    time_interval=1, inc_mode='random', inc_params=None,
                    override_lan=0., nsamp='all', out_fname=None,
                    standard_arr_size=False, posterior_type='radvel'):
    """Generate a point cloud from a posterior DataFrame."""
    if nsamp == 'all':
        nsamp = len(post_df)
        print(f"Using all {nsamp} posterior samples")

    if params is None:
        params = orbit_params[planet]

    print()
    print("-" * 60)
    print(f"Configuration:")
    print(f"  Planet: {display_name(planet)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Time interval: {time_interval} days")

    if posterior_type == 'orbitize':
        print("  Inclination: from Orbitize posterior (sampled)")
        print("  Omega: from Orbitize posterior (sampled)")
        override_inc = None
        user_inc_mean = None
        user_inc_sig = None
    elif inc_mode == 'user_gaussian':
        inc_value, inc_uncertainty = inc_params
        print(f"  Inclination: Gaussian (mu={inc_value:.1f}, "
              f"sigma={inc_uncertainty:.1f}) [user-defined]")
        override_inc = None
        user_inc_mean = inc_value
        user_inc_sig = inc_uncertainty
    elif inc_mode == 'gaussian':
        if "inc_mean" in params and "inc_sig" in params:
            print(f"  Inclination: Gaussian (mu={params['inc_mean']:.1f}, "
                  f"sigma={params['inc_sig']:.1f})")
            override_inc = "gaussian"
            user_inc_mean = None
            user_inc_sig = None
        else:
            print("  No Gaussian priors available. Falling back to random.")
            inc_mode = "random"
            override_inc = None
            user_inc_mean = None
            user_inc_sig = None
    elif inc_mode == 'fixed':
        inc_value = inc_params[0]
        print(f"  Inclination: {inc_value:.1f} deg (fixed)")
        override_inc = inc_value
        user_inc_mean = None
        user_inc_sig = None
    else:
        print("  Inclination: random (uniform with critical inc constraint)")
        override_inc = None
        user_inc_mean = None
        user_inc_sig = None

    print(f"  Posterior samples: {nsamp}")
    print("-" * 60)
    print()

    t_start = Time(start_date)
    t_end = Time(end_date)
    if t_end <= t_start:
        raise ValueError("End date must be after start date")

    print(f"Sampling {nsamp} orbits from posterior...")
    df_sample = post_df.sample(nsamp, replace=True)

    n_epochs = int((t_end.mjd - t_start.mjd) / time_interval) + 1
    epochs = Time(np.linspace(t_start.mjd, t_end.mjd, n_epochs), format="mjd")

    print(f"Generating point cloud for {n_epochs} epochs...")

    seps_mas, raoff_mas, deoff_mas, m_pl, inc, true_anomaly, z_au, r_au, parallax = \
        compute_sep(
            df_sample, epochs,
            params.get("basis"), params["m0"], params["m0_err"],
            params["plx"], params["plx_err"],
            params["n_planets"], params["pl_num"],
            override_lan=override_lan, override_inc=override_inc,
            inc_mean=params.get("inc_mean"), inc_sig=params.get("inc_sig"),
            user_inc_mean=user_inc_mean, user_inc_sig=user_inc_sig,
            posterior_type=posterior_type)

    phase_angle_rad = np.arccos(z_au / r_au)
    phase_angle_deg = np.degrees(phase_angle_rad)

    m_pl_mjup = m_pl * (u.M_sun / u.M_jup).to('')
    m_pl_mearth = m_pl * (u.M_sun / u.M_earth).to('')

    # Mass-radius relation
    mass_intervals = np.array([0, 2.04, 95.16, 317.828407, 26635.6863, np.inf])
    C = np.array([0.00346053, -0.06613329, 0.48091861, 1.04956612, -2.84926757])
    S = np.array([0.279, 0.50376436, 0.22725968, 0, 0.881])
    r_pl_rearth = np.zeros_like(m_pl_mearth)
    for i in range(len(mass_intervals) - 1):
        mask = ((m_pl_mearth >= mass_intervals[i])
                & (m_pl_mearth < mass_intervals[i + 1]))
        if np.any(mask):
            r_pl_rearth[mask] = 10 ** (C[i] + S[i] * np.log10(m_pl_mearth[mask]))

    r_pl_rjup = r_pl_rearth * (u.R_earth / u.R_jup).to('')
    inc_deg = np.degrees(inc)

    if posterior_type == 'orbitize':
        if 'chi2' in df_sample.columns:
            lnlike = -df_sample['chi2'].values / 2
        else:
            lnlike = np.zeros(len(df_sample))
    else:
        myBasis = Basis(params["basis"], params["n_planets"])
        df_synth = myBasis.to_synth(df_sample)
        lnlike = df_synth["lnprobability"].values

    epoch_vals = epochs.value

    if standard_arr_size:
        lnlike = np.full_like(seps_mas, lnlike)
        epoch_vals = np.full_like(seps_mas, epoch_vals[:, np.newaxis])
        m_pl_mjup = np.full_like(seps_mas, m_pl_mjup)
        r_pl_rjup = np.full_like(seps_mas, r_pl_rjup)
        inc_deg = np.full_like(seps_mas, inc_deg)

    point_cloud = {
        'epoch_mjd': epoch_vals, 'sep_mas': seps_mas,
        'raoff_mas': raoff_mas, 'deoff_mas': deoff_mas,
        'true_anom_deg': true_anomaly, 'z_au': z_au,
        'orbital_radius_au': r_au, 'phase_angle_deg': phase_angle_deg,
        'm_pl_mjup': m_pl_mjup, 'r_pl_rjup': r_pl_rjup,
        'inc_deg': inc_deg, 'ln_likelihood': lnlike,
        'parallax_mas': parallax,
    }

    if out_fname is None:
        planet_name = planet.replace("_", "")
        output_file = f"{planet_name}_{start_date}_to_{end_date}_PointCloud.pkl"
    else:
        output_file = out_fname.split('.')[0] + '_PointCloud.pkl'
    output_fpath = os.path.join(output_dir, output_file)
    print(f'Saving point cloud to {output_fpath}')
    with open(output_fpath, 'wb') as f:
        pickle.dump(point_cloud, f)

    return point_cloud


# =====================================================================
# Summary CSV generation
# =====================================================================

def gen_summary_csv(planet, point_cloud, output_dir='.', output=None):
    """Generate summary CSV with weighted statistics."""

    m_pl_mjup = point_cloud['m_pl_mjup']
    mass_median = np.median(m_pl_mjup)
    mass_16th = np.percentile(m_pl_mjup, 16)
    mass_84th = np.percentile(m_pl_mjup, 84)

    r_pl_rjup = point_cloud['r_pl_rjup']
    rad_median = np.median(r_pl_rjup)
    rad_16th = np.percentile(r_pl_rjup, 16)
    rad_84th = np.percentile(r_pl_rjup, 84)

    inc_deg = point_cloud['inc_deg']
    inc_median = np.median(inc_deg)
    inc_16th = np.percentile(inc_deg, 16)
    inc_84th = np.percentile(inc_deg, 84)

    print(f"Planet mass: {mass_median:.2f} "
          f"+{mass_84th - mass_median:.2f}/-{mass_median - mass_16th:.2f} M_Jup")
    print(f"Planet radius: {rad_median:.2f} "
          f"+{rad_84th - rad_median:.2f}/-{rad_median - rad_16th:.2f} R_Jup")
    print(f"Inclination: {inc_median:.2f} [{inc_16th:.2f}, {inc_84th:.2f}] deg")
    print()

    if point_cloud['epoch_mjd'].ndim == 2:
        epochs = Time(point_cloud['epoch_mjd'][:, 0], format='mjd')
    else:
        epochs = Time(point_cloud['epoch_mjd'], format='mjd')

    start_date = epochs.iso[0][:10]
    end_date = epochs.iso[-1][:10]

    csv_data_dict = {
        'date_iso': epochs.iso,
        'mjd': epochs.mjd,
        'decimal_year': epochs.decimalyear,
    }

    # Detection probabilities
    for key in ['feasible_det_prob_opt', 'feasible_det_prob_con']:
        if key in point_cloud:
            csv_data_dict[key] = point_cloud[key]

    if 'GB_not_observable' in point_cloud:
        csv_data_dict['GB_not_observable'] = point_cloud['GB_not_observable']
    if 'targ_observable' in point_cloud:
        csv_data_dict['targ_observable'] = point_cloud['targ_observable']

    phase_angle_rad = point_cloud['phase_angle_deg'] * np.pi / 180.0
    lambert_phase = (np.sin(phase_angle_rad) + (np.pi - phase_angle_rad)
                     * np.cos(phase_angle_rad)) / np.pi

    labeled_data = {
        'separation_mas': point_cloud['sep_mas'],
        'orbital_radius_au': point_cloud['orbital_radius_au'],
        'phase_angle_deg': point_cloud['phase_angle_deg'],
        'lambert_phase': lambert_phase,
        'true_anomaly': np.degrees(point_cloud['true_anom_deg']) % 360,
    }
    for key in ['phi_x_a', 'flux_contrast']:
        if key in point_cloud:
            labeled_data[key] = point_cloud[key]

    lnlike = point_cloud['ln_likelihood']
    if lnlike.ndim == 2:
        lnlike = lnlike[0]
    weights = np.exp(lnlike - np.max(lnlike))
    weights /= weights.sum()

    for label, arr in labeled_data.items():
        csv_data_dict[f'{label}_median'] = weighted_percentile(arr, weights, 50)
        csv_data_dict[f'{label}_16th'] = weighted_percentile(arr, weights, 16)
        csv_data_dict[f'{label}_84th'] = weighted_percentile(arr, weights, 84)
        csv_data_dict[f'{label}_2.5th'] = weighted_percentile(arr, weights, 2.5)
        csv_data_dict[f'{label}_97.5th'] = weighted_percentile(arr, weights, 97.5)
        csv_data_dict[f'{label}_mean'] = weighted_mean(arr, weights)
        csv_data_dict[f'{label}_std'] = weighted_std(arr, weights)

    if 'integration_time_hours_opt' in point_cloud:
        if 'integration_time_sample_indices' in point_cloud:
            inttime_indices = point_cloud['integration_time_sample_indices']
            inttime_weights = weights[inttime_indices]
            inttime_weights /= inttime_weights.sum()
        else:
            inttime_weights = weights

        for key in ['integration_time_hours_opt', 'integration_time_hours_con',
                     'integration_time_days_opt', 'integration_time_days_con']:
            if key in point_cloud:
                arr = point_cloud[key]
                csv_data_dict[f'{key}_median'] = weighted_percentile_nan(arr, inttime_weights, 50)
                csv_data_dict[f'{key}_16th'] = weighted_percentile_nan(arr, inttime_weights, 16)
                csv_data_dict[f'{key}_84th'] = weighted_percentile_nan(arr, inttime_weights, 84)
                csv_data_dict[f'{key}_2.5th'] = weighted_percentile_nan(arr, inttime_weights, 2.5)
                csv_data_dict[f'{key}_97.5th'] = weighted_percentile_nan(arr, inttime_weights, 97.5)
                csv_data_dict[f'{key}_mean'] = weighted_mean(arr, inttime_weights)
                csv_data_dict[f'{key}_std'] = weighted_std(arr, inttime_weights)

    csv_data = pd.DataFrame(csv_data_dict)

    if output is None:
        planet_name = planet.replace("_", "")
        output_file = f"{planet_name}_separations_{start_date}_to_{end_date}.csv"
    else:
        output_file = output.split('.')[0] + '.csv'

    output_fpath = os.path.join(output_dir, output_file)
    print(f"Writing output to {output_fpath}...")
    csv_data.to_csv(output_fpath, index=False)
    return csv_data