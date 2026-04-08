
import os
import numpy as np
import astropy.units as u
from tqdm import tqdm

from corgidb.photometry import (
    packagePhotometryData,
    loadPhotometryData,
    get_fsed,
)


BAND_CENTERS_NM = {1: 575, 2: 660, 3: 730, 4: 825}
BAND_BPS_PCT    = {1: 10,  2: 17,  3: 17,  4: 12}

# Star name override for Pi Men
STAR_NAME_OVERRIDES = {
    "pi Men":  "HD 39091",
    "pi_Men":  "HD 39091",
}


def _make_band_grid(lam_nm, bp_pct):
    """Build wavelength grid (ws), step, and bandwidth for one band."""
    half = lam_nm / 1000.0 * bp_pct / 200.0
    center = lam_nm / 1000.0
    ws, wstep = np.linspace(center - half, center + half, 100, retstep=True)
    bw = ws[-1] - ws[0]
    return ws, wstep, bw


# Grid loader (call once at notebook startup)
def load_photometry_grid(
    photdata_file="allphotdata.npz",
    dbfile="AlbedoModels.db",
    star_catalog="stdata_2025-02-25.p",
):
    """
    Load the corgidb atmospheric model grid and star catalog.

    Parameters
    ----------
    photdata_file : str
        Path to the photometry data (.npz).
        Generated from ``dbfile`` if it doesn't exist yet.
    dbfile : str
        Path to the AlbedoModels.db file
    star_catalog : str
        Path to the star catalog pickle file.

    Returns
    -------
    dict
        Keys: 'photinterps', 'feinterp', 'distinterp', 'bands', 'stars'
    """
    import pandas as pd

    if not os.path.exists(photdata_file):
        print(f"Building photometry grid from {dbfile} -> {photdata_file}...")
        packagePhotometryData(dbfile=dbfile, outname=photdata_file)

    print(f"Loading photometry grid from {photdata_file}...")
    photdict = loadPhotometryData(infile=photdata_file)

    print(f"Loading star catalog from {star_catalog}...")
    stars = pd.read_pickle(star_catalog)

    bands = {
        b: _make_band_grid(BAND_CENTERS_NM[b], BAND_BPS_PCT[b])
        for b in BAND_CENTERS_NM
    }

    print(f"Photometry grid loaded: {len(stars)} stars, {len(bands)} bands.")
    for b in sorted(bands.keys()):
        ws, wstep, bw = bands[b]
        print(f"  Band {b}: {ws[0]*1000:.1f}-{ws[-1]*1000:.1f} nm "
              f"(center {BAND_CENTERS_NM[b]} nm, BW {BAND_BPS_PCT[b]}%)")

    return {
        "photinterps": photdict["photinterps"],
        "feinterp":    photdict["feinterp"],
        "distinterp":  photdict["distinterp"],
        "bands":       bands,       # {band_num: (ws, wstep, bw)}
        "stars":       stars,
    }


# ---------------------------------------------------------------------------
# Star lookup
# ---------------------------------------------------------------------------

def _resolve_star(grid, planet_name):
    """
    Look up stellar luminosity and metallicity from the catalog.

    Parameters
    ----------
    grid : dict
        From ``load_photometry_grid()``.
    planet_name : str
        Planet name (e.g. 'eps_Eri_b') or star name.

    Returns
    -------
    star_name : str
    lum_solar : float   (L / L_sun)
    lum_fix : float     (sqrt(L / L_sun), for equivalent-insolation scaling)
    fe : float          ([Fe/H])
    """
    name = planet_name
    # remove planet letter (e.g. eps_Eri_b -> eps_Eri)
    if len(name) >= 3 and name[-2] == "_" and name[-1].isalpha():
        name = name[:-2]
    name = name.replace("_", " ")
    name = STAR_NAME_OVERRIDES.get(name, name)

    row = grid["stars"].query("st_name == @name")
    if row.empty:
        avail = sorted(grid["stars"]["st_name"].unique())[:20]
        raise ValueError(
            f"Star '{name}' (from '{planet_name}') not in catalog. "
            f"Available: {avail}..."
        )

    lum_solar = 10 ** row["lum"].iloc[0]
    fe = row["met"].iloc[0]
    return name, lum_solar, lum_solar**0.5, fe


# Core computation: pphi from the atmospheric model grid
def _compute_pphi_vectorized(
    beta_deg, r_au, fsed_vals,
    fe, lum_fix,
    photinterps, feinterp, distinterp,
    ws, wstep, bw,
):
    """
    Compute band-averaged pphi(alpha) for arrays of (phase angle, orbital radius,
    fsed). Groups samples by discretized (r_bin, fsed) to evaluate the
    interpolators in a batch.

    Parameters
    ----------
    beta_deg : (N,) array   - phase angles [deg]
    r_au     : (N,) array   - orbital radii [AU]
    fsed_vals: (N,) array   - cloud sedimentation param per sample
    fe       : float        - stellar [Fe/H]
    lum_fix  : float        - sqrt(L/L_sun)
    photinterps, feinterp, distinterp : corgidb interpolators
    ws, wstep, bw : wavelength grid, step, bandwidth

    Returns
    -------
    pphi : (N,) array - band-averaged albedo x phase function
    """
    N = len(beta_deg)
    pphi = np.full(N, np.nan)
    fe_bin = float(feinterp(fe))
    r_scaled = r_au / lum_fix
    r_bins = np.array([float(distinterp(rs)) for rs in r_scaled])

    # group by (r_bin, fsed)
    groups = {}
    for i in range(N):
        key = (r_bins[i], fsed_vals[i])
        groups.setdefault(key, []).append(i)

    for (r_bin, fsed), indices in groups.items():
        indices = np.array(indices)
        interp_func = photinterps[fe_bin][r_bin][fsed]
        betas_group = beta_deg[indices]

        try:
            sort_idx = np.argsort(betas_group)
            vals_sorted = interp_func(betas_group[sort_idx], ws)
            vals = vals_sorted[np.argsort(sort_idx)]
            pphi_group = vals.sum(axis=1) * wstep / bw
            pphi_group[np.isinf(pphi_group)] = np.nan
            pphi[indices] = pphi_group
        except Exception:
            # fall back if batch fails
            for idx in indices:
                try:
                    val = interp_func(beta_deg[idx], ws).sum(1) * wstep / bw
                    val = val[0] if hasattr(val, "__len__") else val
                    if np.isinf(val):
                        val = np.nan
                    pphi[idx] = val
                except Exception:
                    pphi[idx] = np.nan

    return pphi


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_contrast_picaso(
    point_cloud, planet_name, band, grid,
    seed=1234, show_progress=True,
):
    """
    Compute flux contrast from the Batalha+2018 atmospheric model grid.

    Cloud assumption: uniform prior over 8 discrete f_sed values from the
    Batalha et al. (2018) grid [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0],
    of which f_sed=0.0 is cloud-free. Each posterior sample draws one fsed
    value that persists across all epochs

    Modifies point_cloud, adding:
        pphi          - albedo x phase function, (n_epochs, n_samples)
        flux_contrast - planet/star flux ratio, same shape
        fsed          - cloud sedimentation values, (n_samples,)

    Parameters
    ----------
    point_cloud : dict
        Must contain 'phase_angle_deg', 'orbital_radius_au', 'r_pl_rjup'.
    planet_name : str
        e.g. 'eps_Eri_b'
    band : int
        Band number (1, 2, 3, or 4).
    grid : dict
        From ``load_photometry_grid()``.
    seed : int
        RNG seed for fsed draws.
    show_progress : bool
        Show tqdm bar over epochs.

    Returns
    -------
    point_cloud : dict
    """
    # star params
    star_name, lum_solar, lum_fix, fe = _resolve_star(grid, planet_name)
    print(f"  Star: {star_name}, [Fe/H]={fe:.2f}, L={lum_solar:.3f} L_sun")

    # band params
    ws, wstep, bw = grid["bands"][band]
    print(f"  Band {band}: {ws[0]*1000:.1f}-{ws[-1]*1000:.1f} nm")

    # point cloud arrays
    phase_deg = point_cloud["phase_angle_deg"]   # (n_epochs, n_samples)
    r_au      = point_cloud["orbital_radius_au"] # (n_epochs, n_samples)
    Rp_rjup   = point_cloud["r_pl_rjup"]         # (n_samples,) or (n_epochs, n_samples)
    n_epochs, n_samples = phase_deg.shape

    # draw fsed once per posterior sample (cloud structure is static)
    rng = np.random.default_rng(seed=seed)
    fsed_per_sample = np.array([get_fsed(rng.uniform()) for _ in range(n_samples)])
    #fsed_per_sample=np.zeros(n_samples)
    point_cloud["fsed"] = fsed_per_sample

    # report fsed distribution
    unique, counts = np.unique(fsed_per_sample, return_counts=True)
    fsed_report = ", ".join([f"{v}:{c}" for v, c in zip(unique, counts)])
    print(f"  fsed distribution: {fsed_report}")
    n_clear = np.sum(fsed_per_sample == 0.0)
    print(f"  Cloud-free fraction: {n_clear}/{n_samples} ({100*n_clear/n_samples:.1f}%)")

    # compute pphi epoch by epoch
    pphi_arr = np.full((n_epochs, n_samples), np.nan)
    epoch_iter = tqdm(range(n_epochs), desc=f"  pphi band {band}",
                      unit="epoch") if show_progress else range(n_epochs)

    for i_epoch in epoch_iter:
        pphi_arr[i_epoch, :] = _compute_pphi_vectorized(
            beta_deg=phase_deg[i_epoch, :],
            r_au=r_au[i_epoch, :],
            fsed_vals=fsed_per_sample,
            fe=fe, lum_fix=lum_fix,
            photinterps=grid["photinterps"],
            feinterp=grid["feinterp"],
            distinterp=grid["distinterp"],
            ws=ws, wstep=wstep, bw=bw,
        )

    point_cloud["pphi"] = pphi_arr

    # flux contrast: pphi * (Rp / r)^2
    Rp_au = Rp_rjup * u.R_jup.to(u.AU)
    if Rp_au.ndim == 1:
        Rp_au = Rp_au[np.newaxis, :]

    point_cloud["flux_contrast"] = pphi_arr * (Rp_au / r_au) ** 2

    valid = np.isfinite(pphi_arr)
    n_valid, n_total = valid.sum(), pphi_arr.size
    print(f"  pphi computed: {n_valid}/{n_total} valid ({100*n_valid/n_total:.1f}%)")
    if n_valid > 0:
        print(f"  pphi range: [{np.nanmin(pphi_arr):.4e}, {np.nanmax(pphi_arr):.4e}]")
        fc = point_cloud["flux_contrast"]
        print(f"  flux contrast range: [{np.nanmin(fc):.4e}, {np.nanmax(fc):.4e}]")

    return point_cloud


# ---------------------------------------------------------------------------
# Lambert  (old code)
def compute_contrast_lambert(point_cloud, band, albedo_dict, albedo_std=0.1):
    """
    Original Lambert + Gaussian albedo contrast.
    baseline.
    """
    albedos = np.random.normal(
        albedo_dict[band], albedo_std, size=point_cloud["sep_mas"].shape,
    )
    phase_rad = point_cloud["phase_angle_deg"] * np.pi / 180.0

    point_cloud["lambert_phase"] = (
        np.sin(phase_rad) + (np.pi - phase_rad) * np.cos(phase_rad)
    ) / np.pi
    point_cloud["phi_x_a"] = point_cloud["lambert_phase"] * albedos
    point_cloud["flux_contrast"] = (
        point_cloud["phi_x_a"]
        * (point_cloud["r_pl_rjup"] * u.R_jup.to(u.AU)
           / point_cloud["orbital_radius_au"]) ** 2
    )
    return point_cloud