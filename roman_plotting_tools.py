import matplotlib.gridspec as gridspec
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit

from roman_orbit_tools import orbit_params, display_name


def compute_orbit_for_plotting(
    df, epochs, basis=None, m0=None, m0_err=None,
    plx=None, plx_err=None, n_planets=1, pl_num=1,
    override_inc=None, override_lan=None,
    inc_mean=None, inc_sig=None,
    user_inc_mean=None, user_inc_sig=None,
    posterior_type='radvel',
):
    """
    Compute orbit trajectories for 2D plotting (RA/Dec offsets).
    Returns the same separation data as compute_sep for plots.
    Supports both RadVel and Orbitize posteriors.

    Args:
        df (pd.DataFrame): Posterior samples
        epochs (astropy.time.Time): Epochs for orbit computation
        basis (str): RadVel basis string (required for RadVel, unused for Orbitize)
        m0 (float): Stellar mass (required for RadVel, optional for Orbitize)
        m0_err (float): Stellar mass error (RadVel only)
        plx (float): Parallax in mas (required for RadVel, optional for Orbitize)
        plx_err (float): Parallax error (RadVel only)
        n_planets (int): Number of planets in system
        pl_num (int): Planet number
        override_inc (float): Override inclination (RadVel only)
        override_lan (float): Override longitude of ascending node (RadVel only)
        inc_mean (float): Mean inclination for Gaussian sampling (RadVel only)
        inc_sig (float): Std dev inclination for Gaussian sampling (RadVel only)
        user_inc_mean (float): User-provided mean inclination (RadVel only)
        user_inc_sig (float): User-provided std dev inclination (RadVel only)
        posterior_type (str): 'radvel' or 'orbitize'

    Returns:
        tuple: (raoff, deoff, best_idx)
            raoff: RA offsets in mas (n_epochs x n_samples)
            deoff: Dec offsets in mas (n_epochs x n_samples)
            best_idx: Index of best-fit orbit
    """
    chain_len = len(df)
    tau_ref_epoch = 58849

    if posterior_type == 'orbitize':
        sma = df[f'sma{pl_num}'].values
        ecc = df[f'ecc{pl_num}'].values
        inc = np.radians(df[f'inc{pl_num}'].values) * 180 / np.pi
        omega_pl_rad = np.radians(df[f'aop{pl_num}'].values) * 180 / np.pi
        lan = np.radians(df[f'pan{pl_num}'].values) * 180 / np.pi
        tau = df[f'tau{pl_num}'].values

        if 'm0' in df.columns:
            m_st = df['m0'].values
        elif m0 is not None:
            m_st = np.full(chain_len, m0)
        else:
            raise ValueError("Need stellar mass (m0)")

        m_pl = (df[f'm{pl_num}'].values
                if f'm{pl_num}' in df.columns else 0.001 * m_st)
        mtot = m_st + m_pl

        if 'plx' in df.columns:
            parallax = df['plx'].values
        elif 'parallax' in df.columns:
            parallax = df['parallax'].values
        elif plx is not None:
            parallax = np.random.normal(
                plx, plx_err if plx_err else 0.01 * plx, size=chain_len)
        else:
            raise ValueError("Need parallax")

        best_idx = (np.argmin(df['chi2'].values)
                    if 'chi2' in df.columns else 0)
    else:
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
            inc = np.radians(np.clip(
                np.random.normal(user_inc_mean, user_inc_sig, size=chain_len),
                0, 180))
        elif override_inc is not None and override_inc != "gaussian":
            inc = np.full(chain_len, np.radians(override_inc))
        elif (override_inc == "gaussian"
              and inc_mean is not None and inc_sig is not None):
            inc = np.radians(np.clip(
                np.random.normal(inc_mean, inc_sig, size=chain_len), 0, 180))
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
        best_idx = np.argmax(df["lnprobability"].values)

    raoff, deoff, vz = calc_orbit(
        epochs.mjd, sma, ecc, inc,
        omega_pl_rad, lan, tau,
        parallax, mtot, tau_ref_epoch=tau_ref_epoch)

    return raoff, deoff, best_idx



# =====================================================================
# Orbit trajectory helper (for 2-D orbit panel)
# =====================================================================

def compute_orbit_for_plotting(
    df, epochs, basis=None, m0=None, m0_err=None,
    plx=None, plx_err=None, n_planets=1, pl_num=1,
    override_inc=None, override_lan=None,
    inc_mean=None, inc_sig=None,
    user_inc_mean=None, user_inc_sig=None,
    posterior_type='radvel',
):
    """Compute RA/Dec offsets for orbit trajectory visualization."""
    chain_len = len(df)
    tau_ref_epoch = 58849

    if posterior_type == 'orbitize':
        sma = df[f'sma{pl_num}'].values
        ecc = df[f'ecc{pl_num}'].values
        inc = np.radians(df[f'inc{pl_num}'].values) * 180 / np.pi
        omega_pl_rad = np.radians(df[f'aop{pl_num}'].values) * 180 / np.pi
        lan = np.radians(df[f'pan{pl_num}'].values) * 180 / np.pi
        tau = df[f'tau{pl_num}'].values

        if 'm0' in df.columns:
            m_st = df['m0'].values
        elif m0 is not None:
            m_st = np.full(chain_len, m0)
        else:
            raise ValueError("Need stellar mass (m0)")

        m_pl = (df[f'm{pl_num}'].values
                if f'm{pl_num}' in df.columns else 0.001 * m_st)
        mtot = m_st + m_pl

        if 'plx' in df.columns:
            parallax = df['plx'].values
        elif 'parallax' in df.columns:
            parallax = df['parallax'].values
        elif plx is not None:
            parallax = np.random.normal(
                plx, plx_err if plx_err else 0.01 * plx, size=chain_len)
        else:
            raise ValueError("Need parallax")

        best_idx = (np.argmin(df['chi2'].values)
                    if 'chi2' in df.columns else 0)
    else:
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
            inc = np.radians(np.clip(
                np.random.normal(user_inc_mean, user_inc_sig, size=chain_len),
                0, 180))
        elif override_inc is not None and override_inc != "gaussian":
            inc = np.full(chain_len, np.radians(override_inc))
        elif (override_inc == "gaussian"
              and inc_mean is not None and inc_sig is not None):
            inc = np.radians(np.clip(
                np.random.normal(inc_mean, inc_sig, size=chain_len), 0, 180))
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
        best_idx = np.argmax(df["lnprobability"].values)

    raoff, deoff, vz = calc_orbit(
        epochs.mjd, sma, ecc, inc,
        omega_pl_rad, lan, tau,
        parallax, mtot, tau_ref_epoch=tau_ref_epoch)

    return raoff, deoff, best_idx


# =====================================================================
# Time-series orbital parameter plots
# =====================================================================

def plot_orbital_parameters(
    planet, csv_data, output_prefix,
    df_sample=None, params=None, override_inc=None,
    override_lan=None, user_inc_mean=None, user_inc_sig=None,
    start_date=None, end_date=None, figsize=None, fig_ext='png',
    band=None, IWA_mas=None, OWA_mas=None,
    posterior_type='radvel', show_WFOV=False, show_plots=False,
):
    """Create time-series parameter plots with optional 2D orbit panel."""

    years = csv_data['decimal_year'].values

    cm = plt.cm.plasma
    c_median = cm(0.6)
    c_fill_68 = cm(0.2)
    c_fill_95 = cm(0.15)
    c_iwa_narrow = cm(0.85)
    c_iwa_wide = cm(0.5)
    c_orbit_light = cm(0.2)
    c_star = cm(0.0)
    c_opt = cm(0.3)
    c_con = cm(0.7)

    if IWA_mas is not None and OWA_mas is not None:
        IWA, OWA = IWA_mas, OWA_mas
    elif show_WFOV:
        IWA, OWA = 300, 994
    else:
        IWA, OWA = 155, 436

    iwa_color = c_iwa_wide if show_WFOV else c_iwa_narrow
    iwa_label = f'IWA/OWA ({"Wide" if show_WFOV else "Narrow"})'

    plot_2d = (df_sample is not None and params is not None)

    # Count panels
    n_param_plots = 2  # separation + phase angle
    sep, phase = 0, 1

    has_feasible = 'feasible_det_prob_opt' in csv_data.columns
    plot_det = has_feasible
    if plot_det:
        n_param_plots += 1
        sep += 1
        phase += 1
        det = 0

    plot_fc = 'flux_contrast_median' in csv_data.columns
    if plot_fc:
        n_param_plots += 1
        fc = phase + 1

    plot_inttime = ('integration_time_hours_opt_median' in csv_data.columns
                    and 'integration_time_hours_con_median' in csv_data.columns)
    if plot_inttime:
        n_param_plots += 1
        inttime_idx = n_param_plots - 1

    if plot_2d:
        if figsize is None:
            figsize = (20, 12)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_param_plots, 2, width_ratios=[1.2, 1],
                              hspace=0.3, wspace=0.3)

        epochs_2d = Time(np.linspace(
            Time(start_date).mjd, Time(end_date).mjd, 100), format="mjd")
        raoff_2d, deoff_2d, best_idx = compute_orbit_for_plotting(
            df_sample, epochs_2d,
            params.get("basis"), params["m0"], params["m0_err"],
            params["plx"], params["plx_err"],
            params["n_planets"], params["pl_num"],
            override_inc=override_inc, override_lan=override_lan,
            inc_mean=params.get("inc_mean"), inc_sig=params.get("inc_sig"),
            user_inc_mean=user_inc_mean, user_inc_sig=user_inc_sig,
            posterior_type=posterior_type)

        ax_orbit = fig.add_subplot(gs[:, 0])

        if posterior_type == 'orbitize':
            inc_col = f'inc{params["pl_num"]}'
            if inc_col in df_sample.columns:
                inc_med = np.median(df_sample[inc_col]) * 180 / np.pi
                inc_str = f'{inc_med:.1f} deg'
            else:
                inc_str = 'from posterior'
            lan_str = 'from posterior'
        else:
            inc_str = ('random' if override_inc is None
                       else f'{override_inc} deg')
            lan_str = ('random' if override_lan is None
                       else f'{override_lan} deg')

        ax_orbit.set_title(
            f'{display_name(planet)}: Orbital Trajectory\n'
            f'(i={inc_str}, Ω={lan_str})',
            fontsize=14, fontweight='bold', pad=15)
        ax_orbit.set_xlabel('RA Offset [mas]', fontsize=13, fontweight='bold')
        ax_orbit.set_ylabel('Dec Offset [mas]', fontsize=13, fontweight='bold')

        theta = np.linspace(0, 2 * np.pi, 100)
        ax_orbit.plot(IWA * np.cos(theta), IWA * np.sin(theta),
                      color=iwa_color, lw=3, ls='--',
                      label=iwa_label, alpha=0.7)
        ax_orbit.plot(OWA * np.cos(theta), OWA * np.sin(theta),
                      color=iwa_color, lw=3, ls='--', alpha=0.7)

        n_samp = min(50, raoff_2d.shape[1])
        samp_idx = np.random.choice(raoff_2d.shape[1], n_samp, replace=False)
        for i in samp_idx:
            ax_orbit.plot(raoff_2d[:, i], deoff_2d[:, i], '-',
                          color=c_orbit_light, alpha=0.2, lw=1.5)

        ax_orbit.plot(0, 0, '*', color=c_star, markersize=25, label='Star',
                      zorder=15, markeredgecolor='yellow', markeredgewidth=0.5)
        ax_orbit.set_xlim(-1.1 * OWA, 1.1 * OWA)
        ax_orbit.set_ylim(-1.1 * OWA, 1.1 * OWA)
        ax_orbit.set_aspect('equal')
        ax_orbit.grid(True, alpha=0.2, ls=':')
        ax_orbit.legend(loc='best', fontsize=11, framealpha=0.9)

        axes = [fig.add_subplot(gs[i, 1]) for i in range(n_param_plots)]
    else:
        if figsize is None:
            figsize = (14, 12)
        fig, axes = plt.subplots(n_param_plots, 1, figsize=figsize)
        if n_param_plots == 1:
            axes = [axes]

    band_str = f' Band {band}' if band is not None else ''
    fig.suptitle(
        f'{display_name(planet)}{band_str} '
        f'({years[0]:.1f} -> {years[-1]:.1f})',
        fontsize=16, fontweight='bold', y=0.995)

    # ---- Detection Probability ----
    if plot_det:
        ax_dp = axes[det]
        ax_dp.plot(years, csv_data['feasible_det_prob_opt'], '-',
                   color=c_opt, lw=2.5, marker='o', markersize=3,
                   label='Feasible (opt)')
        ax_dp.plot(years, csv_data['feasible_det_prob_con'], '-',
                   color=c_con, lw=2.5, marker='o', markersize=3,
                   label='Feasible (con)')
        ax_dp.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_dp.set_ylabel('Detection Probability', fontsize=11, fontweight='bold')
        ax_dp.set_ylim(0, 1)
        ax_dp.grid(True, alpha=0.25, ls=':')
        ax_dp.tick_params(labelsize=10)

    # ---- Separation ----
    ax1 = axes[sep]
    min_sep = csv_data['separation_mas_16th'].min()
    max_sep = csv_data['separation_mas_84th'].max()
    ax1.set_title(
        f'Angular Separation (1σ: {min_sep:.0f}-{max_sep:.0f} mas)',
        fontsize=12, pad=10)
    ax1.fill_between(years, csv_data['separation_mas_2.5th'],
                     csv_data['separation_mas_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax1.fill_between(years, csv_data['separation_mas_16th'],
                     csv_data['separation_mas_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax1.plot(years, csv_data['separation_mas_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=3)
    ax1.axhline(y=IWA, color=iwa_color, ls='--', lw=2.5,
                label=iwa_label, alpha=0.6)
    ax1.axhline(y=OWA, color=iwa_color, ls='--', lw=2.5, alpha=0.6)
    ax1.set_ylabel('Separation (mas)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, OWA * 1.1)
    ax1.grid(True, alpha=0.25, ls=':')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.tick_params(labelsize=10)

    # ---- Phase Angle ----
    ax3 = axes[phase]
    ax3.fill_between(years, csv_data['phase_angle_deg_2.5th'],
                     csv_data['phase_angle_deg_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax3.fill_between(years, csv_data['phase_angle_deg_16th'],
                     csv_data['phase_angle_deg_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax3.plot(years, csv_data['phase_angle_deg_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=3)
    ax3.set_ylabel('Phase Angle (deg)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 180)
    ax3.grid(True, alpha=0.25, ls=':')
    ax3.legend(loc='best', fontsize=9, framealpha=0.9)
    ax3.tick_params(labelsize=10)

    # ---- Flux Contrast ----
    if plot_fc:
        ax_fc = axes[fc]
        ax_fc.fill_between(years, csv_data['flux_contrast_2.5th'],
                           csv_data['flux_contrast_97.5th'],
                           color=c_fill_95, alpha=0.3, label='95% CI')
        ax_fc.fill_between(years, csv_data['flux_contrast_16th'],
                           csv_data['flux_contrast_84th'],
                           color=c_fill_68, alpha=0.5, label='68% CI')
        ax_fc.plot(years, csv_data['flux_contrast_median'], '-',
                   color=c_median, lw=2.5, label='Median',
                   marker='o', markersize=3)
        ax_fc.set_yscale('log')
        ax_fc.set_ylabel('Flux Contrast', fontsize=11, fontweight='bold')
        ax_fc.set_ylim(1e-9, 1e-7)
        ax_fc.grid(True, alpha=0.25, ls=':')
        ax_fc.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_fc.tick_params(labelsize=10)

    # ---- Integration Time ----
    if plot_inttime:
        ax_int = axes[inttime_idx]
        valid_opt = np.isfinite(
            csv_data['integration_time_hours_opt_median'].values)
        valid_con = np.isfinite(
            csv_data['integration_time_hours_con_median'].values)

        if np.any(valid_opt) or np.any(valid_con):
            for valid_mask, prefix, c, lbl in [
                (valid_opt, 'integration_time_hours_opt', c_opt, 'Optimistic'),
                (valid_con, 'integration_time_hours_con', c_con, 'Conservative'),
            ]:
                if not np.any(valid_mask):
                    continue
                y_med = csv_data[f'{prefix}_median'].values.copy()
                y16 = csv_data[f'{prefix}_16th'].values.copy()
                y84 = csv_data[f'{prefix}_84th'].values.copy()
                y025 = csv_data[f'{prefix}_2.5th'].values.copy()
                y975 = csv_data[f'{prefix}_97.5th'].values.copy()
                for arr in (y_med, y16, y84, y025, y975):
                    arr[~valid_mask] = np.nan
                ax_int.fill_between(years, y025, y975, color=c, alpha=0.15,
                                    label=f'{lbl} 95% CI')
                ax_int.fill_between(years, y16, y84, color=c, alpha=0.30,
                                    label=f'{lbl} 68% CI')
                ax_int.plot(years, y_med, '-', color=c, lw=2.5,
                            label=f'{lbl} median', marker='o', markersize=3)

            ax_int.set_ylabel('Integration Time (hours)',
                              fontsize=11, fontweight='bold')
            ax_int.set_yscale('log')
            all_valid = []
            if np.any(valid_opt):
                all_valid.extend(
                    csv_data['integration_time_hours_opt_median']
                    .values[valid_opt])
            if np.any(valid_con):
                all_valid.extend(
                    csv_data['integration_time_hours_con_median']
                    .values[valid_con])
            if all_valid:
                ax_int.set_ylim(
                    np.nanmin(all_valid) * 0.5, np.nanmax(all_valid) * 2.0)
            ax_int.grid(True, alpha=0.25, ls=':')
            ax_int.legend(loc='best', fontsize=9, framealpha=0.9)
            ax_int.tick_params(labelsize=10)
        else:
            ax_int.text(0.5, 0.5, 'No integration time data available',
                        ha='center', va='center', transform=ax_int.transAxes,
                        fontsize=12, color='gray')
            ax_int.set_ylabel('Integration Time (hours)',
                              fontsize=11, fontweight='bold')

    # ---- Observation windows ----
    for a, ax in enumerate(axes):
        ylims = ax.get_ylim()
        if 'GB_not_observable' in csv_data.columns:
            ax.fill_between(
                years, ylims[0], ylims[1],
                where=~csv_data.GB_not_observable,
                alpha=0.15, edgecolor='None', color='orange',
                label='GB Observations' if a == 0 else '', zorder=0)
        if 'targ_observable' in csv_data.columns:
            ax.fill_between(
                years, ylims[0], ylims[1],
                where=~csv_data.targ_observable,
                alpha=0.15, edgecolor='None', color='gray',
                label='Solar Keepout' if a == 0 else '', zorder=0)
        ax.set_ylim(ylims)
        if a == 0 and ('GB_not_observable' in csv_data.columns
                       or 'targ_observable' in csv_data.columns):
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.set_xlim(years[0], years[-1])

    axes[-1].set_xlabel('Year', fontsize=11, fontweight='bold')
    plt.tight_layout()

    plot_filename = f"{output_prefix}_orbital_params.{fig_ext}"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    if show_plots:
        plt.show()
    plt.close('all')


# =====================================================================
# Contrast comparison mega-plot
# =====================================================================

def plot_contrast_comparison(
    planet, band, csv_data, scenarios, int_times_hr,
    IWA_mas, OWA_mas,
    output_dir, fname_prefix,
    targ_observable=None, GB_not_observable=None,
    contrast_penalty=2.0, target_snr=5, n_zodis=1,
    show_wfov=False, show_plots=True, fig_ext='png',
):
    """Multi-panel mega-plot: pdet, separation, phase, contrast, inttime."""

    years = csv_data['decimal_year'].values
    cm = plt.cm.plasma
    c_median, c_fill_68, c_fill_95 = cm(0.6), cm(0.2), cm(0.15)
    c_opt, c_con = cm(0.3), cm(0.7)

    cc_colors = {
        (int_times_hr[0], 'opt'): cm(0.15),
        (int_times_hr[0], 'con'): cm(0.40),
        (int_times_hr[1], 'opt'): cm(0.65),
        (int_times_hr[1], 'con'): cm(0.90),
    }
    iwa_color = cm(0.5) if show_wfov else cm(0.85)
    iwa_label = f'IWA/OWA ({"Wide" if show_wfov else "Narrow"})'

    has_inttime = ('integration_time_hours_opt_median' in csv_data.columns
                   and 'integration_time_hours_con_median' in csv_data.columns)

    n_panels = 4 + int(has_inttime)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)

    fig.suptitle(
        f'{display_name(planet)} — Band {band}  |  '
        f'Det Prob vs Integration Time Budget  '
        f'({years[0]:.1f} → {years[-1]:.1f})',
        fontsize=16, fontweight='bold', y=1.0)

    ax_idx = 0

    # ---- Detection Probability ----
    ax = axes[ax_idx]; ax_idx += 1
    for t_hr in int_times_hr:
        for scenario in ['opt', 'con']:
            key = f'{t_hr}h {scenario}'
            if key in scenarios:
                ax.plot(years, scenarios[key], '-',
                        color=cc_colors[(t_hr, scenario)], lw=2.5,
                        label=f'{t_hr}h {scenario}')
        deg_key = f'{t_hr}h opt {contrast_penalty:.0f}x'
        if deg_key in scenarios:
            ax.plot(years, scenarios[deg_key], '--',
                    color=cc_colors[(t_hr, 'opt')], lw=2.0, alpha=0.7,
                    label=f'{t_hr}h opt {contrast_penalty:.0f}× deg')
    ax.set_ylabel('Detection Probability', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Detection Probability (feasible inttime, SNR={target_snr}, '
                 f'{n_zodis} zodis)', fontsize=12, pad=8)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, ls=':')

    # ---- Separation ----
    ax1 = axes[ax_idx]; ax_idx += 1
    min_sep = csv_data['separation_mas_16th'].min()
    max_sep = csv_data['separation_mas_84th'].max()
    ax1.set_title(f'Angular Separation (1σ: {min_sep:.0f}–{max_sep:.0f} mas)',
                  fontsize=12, pad=8)
    ax1.fill_between(years, csv_data['separation_mas_2.5th'],
                     csv_data['separation_mas_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax1.fill_between(years, csv_data['separation_mas_16th'],
                     csv_data['separation_mas_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax1.plot(years, csv_data['separation_mas_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=2)
    ax1.axhline(y=IWA_mas, color=iwa_color, ls='--', lw=2.5,
                label=iwa_label, alpha=0.6)
    ax1.axhline(y=OWA_mas, color=iwa_color, ls='--', lw=2.5, alpha=0.6)
    ax1.set_ylabel('Separation (mas)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(OWA_mas * 1.1, max_sep * 1.1))
    ax1.legend(loc='best', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25, ls=':')

    # ---- Phase Angle ----
    ax2 = axes[ax_idx]; ax_idx += 1
    ax2.fill_between(years, csv_data['phase_angle_deg_2.5th'],
                     csv_data['phase_angle_deg_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax2.fill_between(years, csv_data['phase_angle_deg_16th'],
                     csv_data['phase_angle_deg_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax2.plot(years, csv_data['phase_angle_deg_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=2)
    ax2.set_ylabel('Phase Angle (deg)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 180)
    ax2.legend(loc='best', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.25, ls=':')

    # ---- Flux Contrast ----
    ax3 = axes[ax_idx]; ax_idx += 1
    if 'flux_contrast_median' in csv_data.columns:
        ax3.fill_between(years, csv_data['flux_contrast_2.5th'],
                         csv_data['flux_contrast_97.5th'],
                         color=c_fill_95, alpha=0.3, label='95% CI')
        ax3.fill_between(years, csv_data['flux_contrast_16th'],
                         csv_data['flux_contrast_84th'],
                         color=c_fill_68, alpha=0.5, label='68% CI')
        ax3.plot(years, csv_data['flux_contrast_median'], '-',
                 color=c_median, lw=2.5, label='Median',
                 marker='o', markersize=2)
        ax3.set_yscale('log')
        ax3.set_ylabel('Flux Contrast', fontsize=11, fontweight='bold')
        ax3.set_ylim(1e-11, 1e-7)
        ax3.legend(loc='best', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.25, ls=':')

    # ---- Integration Time ----
    if has_inttime:
        ax_int = axes[ax_idx]; ax_idx += 1
        for lbl, prefix, c in [
            ('Optimistic', 'integration_time_hours_opt', c_opt),
            ('Conservative', 'integration_time_hours_con', c_con),
        ]:
            valid = np.isfinite(csv_data[f'{prefix}_median'].values)
            if np.any(valid):
                y_med = csv_data[f'{prefix}_median'].values.copy()
                y16 = csv_data[f'{prefix}_16th'].values.copy()
                y84 = csv_data[f'{prefix}_84th'].values.copy()
                y025 = csv_data[f'{prefix}_2.5th'].values.copy()
                y975 = csv_data[f'{prefix}_97.5th'].values.copy()
                for arr in (y_med, y16, y84, y025, y975):
                    arr[~valid] = np.nan
                ax_int.fill_between(years, y025, y975, color=c, alpha=0.15,
                                    label=f'{lbl} 95% CI')
                ax_int.fill_between(years, y16, y84, color=c, alpha=0.30,
                                    label=f'{lbl} 68% CI')
                ax_int.plot(years, y_med, '-', color=c, lw=2.5,
                            label=f'{lbl} median', marker='o', markersize=2)
        ax_int.set_yscale('log')
        ax_int.set_ylabel('Integration Time (hours)',
                          fontsize=11, fontweight='bold')
        ax_int.legend(loc='best', fontsize=8, framealpha=0.9)
        ax_int.grid(True, alpha=0.25, ls=':')

    # ---- Observation windows ----
    for i, ax in enumerate(axes):
        ylims = ax.get_ylim()
        if targ_observable is not None:
            ax.fill_between(
                years, ylims[0], ylims[1], where=~targ_observable,
                alpha=0.12, color='gray',
                label='Solar keepout' if i == 0 else '', zorder=0)
        if GB_not_observable is not None:
            ax.fill_between(
                years, ylims[0], ylims[1], where=~GB_not_observable,
                alpha=0.12, color='orange',
                label='GB obs' if i == 0 else '', zorder=0)
        ax.set_ylim(ylims)
        ax.set_xlim(years[0], years[-1])
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel('Year', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plot_fpath = os.path.join(
        output_dir,
        f'{fname_prefix}_Band{band}_inttime_contrast_comparison.{fig_ext}')
    plt.savefig(plot_fpath, dpi=150, bbox_inches='tight')
    print(f'  Plot saved to {plot_fpath}')
    if show_plots:
        plt.show()
    plt.close('all')


# =====================================================================
# Integration time histogram at a specific epoch
# =====================================================================

def plot_inttime_histogram(
    planet, band, point_cloud,
    target_date=None, date_range=None,
    scenario='opt', n_zodis=1,
    percentiles=None, n_bins=35,
    title=None, show_plots=True, save_path=None,
):
    """Plot weighted integration time distribution at a single epoch.

    Either pick a specific date, or find the epoch with the highest
    detection probability (most valid draws) within a date range.

    Parameters
    ----------
    planet : str
    band : int
    point_cloud : dict
        Must contain integration_time_hours_opt/con and
        integration_time_sample_indices.
    target_date : str, optional
        ISO date (e.g. '2027-01-15'). Picks the nearest epoch.
    date_range : tuple of str, optional
        (start, end) ISO dates. Picks the epoch with the most valid
        integration time draws in this range. Ignored if target_date is set.
    scenario : str
        'opt' or 'con'.
    n_zodis : int
        For the plot title only.
    percentiles : list of int, optional
        Percentiles to mark. Default [50, 80, 95].
    n_bins : int
        Number of histogram bins.
    title : str, optional
        Override the auto-generated title.
    show_plots : bool
    save_path : str, optional
        Save figure to this path.

    Returns
    -------
    dict with keys: epoch_date, median_h, pctls, n_valid, n_total
    """
    if percentiles is None:
        percentiles = [50, 80, 95]

    # Epoch array
    if point_cloud['epoch_mjd'].ndim == 2:
        epoch_mjd = point_cloud['epoch_mjd'][:, 0]
    else:
        epoch_mjd = point_cloud['epoch_mjd']

    # Find the epoch
    if target_date is not None:
        target_mjd = Time(target_date).mjd
        i_epoch = int(np.argmin(np.abs(epoch_mjd - target_mjd)))
    elif date_range is not None:
        mjd_start = Time(date_range[0]).mjd
        mjd_end = Time(date_range[1]).mjd
        mask = (epoch_mjd >= mjd_start) & (epoch_mjd <= mjd_end)
        if not np.any(mask):
            print(f"No epochs in range {date_range}")
            return None

        key = f'integration_time_hours_{scenario}'
        arr = point_cloud[key]
        indices = np.where(mask)[0]

        # Pick epoch with most valid draws (= peak det prob)
        n_valid_per_epoch = np.sum(np.isfinite(arr[indices, :]), axis=1)
        i_epoch = indices[np.argmax(n_valid_per_epoch)]
    else:
        raise ValueError("Provide either target_date or date_range")

    actual_date = Time(epoch_mjd[i_epoch], format='mjd').iso[:10]

    # Weights
    sample_indices = point_cloud['integration_time_sample_indices']
    lnlike = point_cloud['ln_likelihood']
    if lnlike.ndim == 2:
        lnlike = lnlike[0]
    w = np.exp(lnlike - np.max(lnlike))
    w /= w.sum()
    inttime_w = w[sample_indices]
    inttime_w /= inttime_w.sum()

    # Integration times at this epoch
    key = f'integration_time_hours_{scenario}'
    t_hrs = point_cloud[key][i_epoch, :]
    valid = np.isfinite(t_hrs)
    n_valid = int(valid.sum())
    n_total = len(t_hrs)

    if n_valid < 5:
        print(f"Only {n_valid} valid samples at {actual_date} — skipping.")
        return None

    t_valid = t_hrs[valid]
    w_valid = inttime_w[valid]
    w_valid /= w_valid.sum()

    # Weighted percentiles
    sorted_idx = np.argsort(t_valid)
    cumw = np.cumsum(w_valid[sorted_idx])
    pctls = {}
    for pct in percentiles:
        idx = np.searchsorted(cumw, pct / 100)
        if idx < len(t_valid):
            pctls[pct] = t_valid[sorted_idx[idx]]

    # Plot
    cm = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.logspace(
        np.log10(max(t_valid.min(), 0.1)),
        np.log10(t_valid.max()), n_bins)
    ax.hist(t_valid, bins=bins, weights=w_valid, alpha=0.85,
            color=cm(0.25), edgecolor='white', linewidth=0.5)

    pctl_styles = {
        50: ('-',  2.5, 'Median'),
        80: ('--', 2.0, '80th pctl'),
        95: (':', 1.8, '95th pctl'),
    }
    for pct in percentiles:
        if pct in pctls:
            ls, lw, name = pctl_styles.get(pct, ('--', 1.5, f'{pct}th pctl'))
            ax.axvline(pctls[pct], color=cm(0.7), ls=ls, lw=lw,
                       label=f'{name}: {pctls[pct]:.1f}h')

    ax.set_xscale('log')
    ax.set_xlabel('Integration Time (hours)', fontsize=12)
    ax.set_ylabel('Weighted Probability', fontsize=12)

    scen_label = 'Optimistic' if scenario == 'opt' else 'Conservative'
    if title is None:
        title = (f'{display_name(planet)} — Band {band}, '
                 f'{n_zodis} zodi, {scen_label}\n'
                 f'Integration Time Distribution at {actual_date}')
        if date_range is not None:
            title += f'\n(peak det prob epoch in {date_range[0]} to {date_range[1]})'
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle=':')
    ax.set_xlim(bins[0] * 0.9, bins[-1] * 1.1)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    if show_plots:
        plt.show()
    plt.close('all')

    return {
        'epoch_date': actual_date,
        'median_h': pctls.get(50),
        'pctls': pctls,
        'n_valid': n_valid,
        'n_total': n_total,
    }
# =====================================================================
# Time-series orbital parameter plots
# =====================================================================

def plot_orbital_parameters(
    planet, csv_data, output_prefix,
    df_sample=None, params=None, override_inc=None,
    override_lan=None, user_inc_mean=None, user_inc_sig=None,
    start_date=None, end_date=None, figsize=None, fig_ext='png',
    band=None, IWA_mas=None, OWA_mas=None,
    posterior_type='radvel', show_WFOV=False, show_plots=False,
):
    """Create time-series parameter plots with optional 2D orbit panel."""

    years = csv_data['decimal_year'].values

    cm = plt.cm.plasma
    c_median = cm(0.6)
    c_fill_68 = cm(0.2)
    c_fill_95 = cm(0.15)
    c_iwa_narrow = cm(0.85)
    c_iwa_wide = cm(0.5)
    c_orbit_light = cm(0.2)
    c_star = cm(0.0)
    c_opt = cm(0.3)
    c_con = cm(0.7)

    if IWA_mas is not None and OWA_mas is not None:
        IWA, OWA = IWA_mas, OWA_mas
    elif show_WFOV:
        IWA, OWA = 300, 994
    else:
        IWA, OWA = 155, 436

    iwa_color = c_iwa_wide if show_WFOV else c_iwa_narrow
    iwa_label = f'IWA/OWA ({"Wide" if show_WFOV else "Narrow"})'

    plot_2d = (df_sample is not None and params is not None)

    # Count panels
    n_param_plots = 2  # separation + phase angle
    sep, phase = 0, 1

    has_feasible = 'feasible_det_prob_opt' in csv_data.columns
    plot_det = has_feasible
    if plot_det:
        n_param_plots += 1
        sep += 1
        phase += 1
        det = 0

    plot_fc = 'flux_contrast_median' in csv_data.columns
    if plot_fc:
        n_param_plots += 1
        fc = phase + 1

    plot_inttime = ('integration_time_hours_opt_median' in csv_data.columns
                    and 'integration_time_hours_con_median' in csv_data.columns)
    if plot_inttime:
        n_param_plots += 1
        inttime_idx = n_param_plots - 1

    if plot_2d:
        if figsize is None:
            figsize = (20, 12)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_param_plots, 2, width_ratios=[1.2, 1],
                              hspace=0.3, wspace=0.3)

        epochs_2d = Time(np.linspace(
            Time(start_date).mjd, Time(end_date).mjd, 100), format="mjd")
        raoff_2d, deoff_2d, best_idx = compute_orbit_for_plotting(
            df_sample, epochs_2d,
            params.get("basis"), params["m0"], params["m0_err"],
            params["plx"], params["plx_err"],
            params["n_planets"], params["pl_num"],
            override_inc=override_inc, override_lan=override_lan,
            inc_mean=params.get("inc_mean"), inc_sig=params.get("inc_sig"),
            user_inc_mean=user_inc_mean, user_inc_sig=user_inc_sig,
            posterior_type=posterior_type)

        ax_orbit = fig.add_subplot(gs[:, 0])

        if posterior_type == 'orbitize':
            inc_col = f'inc{params["pl_num"]}'
            if inc_col in df_sample.columns:
                inc_med = np.median(df_sample[inc_col]) * 180 / np.pi
                inc_str = f'{inc_med:.1f} deg'
            else:
                inc_str = 'from posterior'
            lan_str = 'from posterior'
        else:
            inc_str = ('random' if override_inc is None
                       else f'{override_inc} deg')
            lan_str = ('random' if override_lan is None
                       else f'{override_lan} deg')

        ax_orbit.set_title(
            f'{display_name(planet)}: Orbital Trajectory\n'
            f'(i={inc_str}, Ω={lan_str})',
            fontsize=14, fontweight='bold', pad=15)
        ax_orbit.set_xlabel('RA Offset [mas]', fontsize=13, fontweight='bold')
        ax_orbit.set_ylabel('Dec Offset [mas]', fontsize=13, fontweight='bold')

        theta = np.linspace(0, 2 * np.pi, 100)
        ax_orbit.plot(IWA * np.cos(theta), IWA * np.sin(theta),
                      color=iwa_color, lw=3, ls='--',
                      label=iwa_label, alpha=0.7)
        ax_orbit.plot(OWA * np.cos(theta), OWA * np.sin(theta),
                      color=iwa_color, lw=3, ls='--', alpha=0.7)

        n_samp = min(50, raoff_2d.shape[1])
        samp_idx = np.random.choice(raoff_2d.shape[1], n_samp, replace=False)
        for i in samp_idx:
            ax_orbit.plot(raoff_2d[:, i], deoff_2d[:, i], '-',
                          color=c_orbit_light, alpha=0.2, lw=1.5)

        ax_orbit.plot(0, 0, '*', color=c_star, markersize=25, label='Star',
                      zorder=15, markeredgecolor='yellow', markeredgewidth=0.5)
        ax_orbit.set_xlim(-1.1 * OWA, 1.1 * OWA)
        ax_orbit.set_ylim(-1.1 * OWA, 1.1 * OWA)
        ax_orbit.set_aspect('equal')
        ax_orbit.grid(True, alpha=0.2, ls=':')
        ax_orbit.legend(loc='best', fontsize=11, framealpha=0.9)

        axes = [fig.add_subplot(gs[i, 1]) for i in range(n_param_plots)]
    else:
        if figsize is None:
            figsize = (14, 12)
        fig, axes = plt.subplots(n_param_plots, 1, figsize=figsize)
        if n_param_plots == 1:
            axes = [axes]

    band_str = f' Band {band}' if band is not None else ''
    fig.suptitle(
        f'{display_name(planet)}{band_str} '
        f'({years[0]:.1f} -> {years[-1]:.1f})',
        fontsize=16, fontweight='bold', y=0.995)

    # ---- Detection Probability ----
    if plot_det:
        ax_dp = axes[det]
        ax_dp.plot(years, csv_data['feasible_det_prob_opt'], '-',
                   color=c_opt, lw=2.5, marker='o', markersize=3,
                   label='Feasible (opt)')
        ax_dp.plot(years, csv_data['feasible_det_prob_con'], '-',
                   color=c_con, lw=2.5, marker='o', markersize=3,
                   label='Feasible (con)')
        ax_dp.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_dp.set_ylabel('Detection Probability', fontsize=11, fontweight='bold')
        ax_dp.set_ylim(0, 1)
        ax_dp.grid(True, alpha=0.25, ls=':')
        ax_dp.tick_params(labelsize=10)

    # ---- Separation ----
    ax1 = axes[sep]
    min_sep = csv_data['separation_mas_16th'].min()
    max_sep = csv_data['separation_mas_84th'].max()
    ax1.set_title(
        f'Angular Separation (1σ: {min_sep:.0f}-{max_sep:.0f} mas)',
        fontsize=12, pad=10)
    ax1.fill_between(years, csv_data['separation_mas_2.5th'],
                     csv_data['separation_mas_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax1.fill_between(years, csv_data['separation_mas_16th'],
                     csv_data['separation_mas_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax1.plot(years, csv_data['separation_mas_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=3)
    ax1.axhline(y=IWA, color=iwa_color, ls='--', lw=2.5,
                label=iwa_label, alpha=0.6)
    ax1.axhline(y=OWA, color=iwa_color, ls='--', lw=2.5, alpha=0.6)
    ax1.set_ylabel('Separation (mas)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, OWA * 1.1)
    ax1.grid(True, alpha=0.25, ls=':')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.tick_params(labelsize=10)

    # ---- Phase Angle ----
    ax3 = axes[phase]
    ax3.fill_between(years, csv_data['phase_angle_deg_2.5th'],
                     csv_data['phase_angle_deg_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax3.fill_between(years, csv_data['phase_angle_deg_16th'],
                     csv_data['phase_angle_deg_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax3.plot(years, csv_data['phase_angle_deg_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=3)
    ax3.set_ylabel('Phase Angle (deg)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 180)
    ax3.grid(True, alpha=0.25, ls=':')
    ax3.legend(loc='best', fontsize=9, framealpha=0.9)
    ax3.tick_params(labelsize=10)

    # ---- Flux Contrast ----
    if plot_fc:
        ax_fc = axes[fc]
        ax_fc.fill_between(years, csv_data['flux_contrast_2.5th'],
                           csv_data['flux_contrast_97.5th'],
                           color=c_fill_95, alpha=0.3, label='95% CI')
        ax_fc.fill_between(years, csv_data['flux_contrast_16th'],
                           csv_data['flux_contrast_84th'],
                           color=c_fill_68, alpha=0.5, label='68% CI')
        ax_fc.plot(years, csv_data['flux_contrast_median'], '-',
                   color=c_median, lw=2.5, label='Median',
                   marker='o', markersize=3)
        ax_fc.set_yscale('log')
        ax_fc.set_ylabel('Flux Contrast', fontsize=11, fontweight='bold')
        ax_fc.set_ylim(1e-9, 1e-7)
        ax_fc.grid(True, alpha=0.25, ls=':')
        ax_fc.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_fc.tick_params(labelsize=10)

    # ---- Integration Time ----
    if plot_inttime:
        ax_int = axes[inttime_idx]
        valid_opt = np.isfinite(
            csv_data['integration_time_hours_opt_median'].values)
        valid_con = np.isfinite(
            csv_data['integration_time_hours_con_median'].values)

        if np.any(valid_opt) or np.any(valid_con):
            for valid_mask, prefix, c, lbl in [
                (valid_opt, 'integration_time_hours_opt', c_opt, 'Optimistic'),
                (valid_con, 'integration_time_hours_con', c_con, 'Conservative'),
            ]:
                if not np.any(valid_mask):
                    continue
                y_med = csv_data[f'{prefix}_median'].values.copy()
                y16 = csv_data[f'{prefix}_16th'].values.copy()
                y84 = csv_data[f'{prefix}_84th'].values.copy()
                y025 = csv_data[f'{prefix}_2.5th'].values.copy()
                y975 = csv_data[f'{prefix}_97.5th'].values.copy()
                for arr in (y_med, y16, y84, y025, y975):
                    arr[~valid_mask] = np.nan
                ax_int.fill_between(years, y025, y975, color=c, alpha=0.15,
                                    label=f'{lbl} 95% CI')
                ax_int.fill_between(years, y16, y84, color=c, alpha=0.30,
                                    label=f'{lbl} 68% CI')
                ax_int.plot(years, y_med, '-', color=c, lw=2.5,
                            label=f'{lbl} median', marker='o', markersize=3)

            ax_int.set_ylabel('Integration Time (hours)',
                              fontsize=11, fontweight='bold')
            ax_int.set_yscale('log')
            all_valid = []
            if np.any(valid_opt):
                all_valid.extend(
                    csv_data['integration_time_hours_opt_median']
                    .values[valid_opt])
            if np.any(valid_con):
                all_valid.extend(
                    csv_data['integration_time_hours_con_median']
                    .values[valid_con])
            if all_valid:
                ax_int.set_ylim(
                    np.nanmin(all_valid) * 0.5, np.nanmax(all_valid) * 2.0)
            ax_int.grid(True, alpha=0.25, ls=':')
            ax_int.legend(loc='best', fontsize=9, framealpha=0.9)
            ax_int.tick_params(labelsize=10)
        else:
            ax_int.text(0.5, 0.5, 'No integration time data available',
                        ha='center', va='center', transform=ax_int.transAxes,
                        fontsize=12, color='gray')
            ax_int.set_ylabel('Integration Time (hours)',
                              fontsize=11, fontweight='bold')

    # ---- Observation windows ----
    for a, ax in enumerate(axes):
        ylims = ax.get_ylim()
        if 'GB_not_observable' in csv_data.columns:
            ax.fill_between(
                years, ylims[0], ylims[1],
                where=~csv_data.GB_not_observable,
                alpha=0.15, edgecolor='None', color='orange',
                label='GB Observations' if a == 0 else '', zorder=0)
        if 'targ_observable' in csv_data.columns:
            ax.fill_between(
                years, ylims[0], ylims[1],
                where=~csv_data.targ_observable,
                alpha=0.15, edgecolor='None', color='gray',
                label='Solar Keepout' if a == 0 else '', zorder=0)
        ax.set_ylim(ylims)
        if a == 0 and ('GB_not_observable' in csv_data.columns
                       or 'targ_observable' in csv_data.columns):
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.set_xlim(years[0], years[-1])

    axes[-1].set_xlabel('Year', fontsize=11, fontweight='bold')
    plt.tight_layout()

    plot_filename = f"{output_prefix}_orbital_params.{fig_ext}"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    if show_plots:
        plt.show()
    plt.close('all')



def plot_contrast_comparison(
    planet, band, csv_data, scenarios, int_times_hr,
    IWA_mas, OWA_mas,
    output_dir, fname_prefix,
    targ_observable=None, GB_not_observable=None,
    contrast_penalty=2.0, target_snr=5, n_zodis=1,
    show_wfov=False, show_plots=True, fig_ext='png',
):
    """Multi-panel megaplot: pdet, separation, phase, contrast, inttime."""

    years = csv_data['decimal_year'].values
    cm = plt.cm.plasma
    c_median, c_fill_68, c_fill_95 = cm(0.6), cm(0.2), cm(0.15)
    c_opt, c_con = cm(0.3), cm(0.7)

    cc_colors = {
        (int_times_hr[0], 'opt'): cm(0.15),
        (int_times_hr[0], 'con'): cm(0.40),
        (int_times_hr[1], 'opt'): cm(0.65),
        (int_times_hr[1], 'con'): cm(0.90),
    }
    iwa_color = cm(0.5) if show_wfov else cm(0.85)
    iwa_label = f'IWA/OWA ({"Wide" if show_wfov else "Narrow"})'

    has_inttime = ('integration_time_hours_opt_median' in csv_data.columns
                   and 'integration_time_hours_con_median' in csv_data.columns)

    n_panels = 4 + int(has_inttime)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)

    fig.suptitle(
        f'{display_name(planet)} — Band {band}  |  '
        f'Det Prob vs Integration Time Budget  '
        f'({years[0]:.1f} → {years[-1]:.1f})',
        fontsize=16, fontweight='bold', y=1.0)

    ax_idx = 0

    # ---- Detection Probability ----
    ax = axes[ax_idx]; ax_idx += 1
    for t_hr in int_times_hr:
        for scenario in ['opt', 'con']:
            key = f'{t_hr}h {scenario}'
            if key in scenarios:
                ax.plot(years, scenarios[key], '-',
                        color=cc_colors[(t_hr, scenario)], lw=2.5,
                        label=f'{t_hr}h {scenario}')
        deg_key = f'{t_hr}h opt {contrast_penalty:.0f}x'
        if deg_key in scenarios:
            ax.plot(years, scenarios[deg_key], '--',
                    color=cc_colors[(t_hr, 'opt')], lw=2.0, alpha=0.7,
                    label=f'{t_hr}h opt {contrast_penalty:.0f}× deg')
    ax.set_ylabel('Detection Probability', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Detection Probability (feasible inttime, SNR={target_snr}, '
                 f'{n_zodis} zodis)', fontsize=12, pad=8)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, ls=':')

    # ---- Separation ----
    ax1 = axes[ax_idx]; ax_idx += 1
    min_sep = csv_data['separation_mas_16th'].min()
    max_sep = csv_data['separation_mas_84th'].max()
    ax1.set_title(f'Angular Separation (1σ: {min_sep:.0f}–{max_sep:.0f} mas)',
                  fontsize=12, pad=8)
    ax1.fill_between(years, csv_data['separation_mas_2.5th'],
                     csv_data['separation_mas_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax1.fill_between(years, csv_data['separation_mas_16th'],
                     csv_data['separation_mas_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax1.plot(years, csv_data['separation_mas_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=2)
    ax1.axhline(y=IWA_mas, color=iwa_color, ls='--', lw=2.5,
                label=iwa_label, alpha=0.6)
    ax1.axhline(y=OWA_mas, color=iwa_color, ls='--', lw=2.5, alpha=0.6)
    ax1.set_ylabel('Separation (mas)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(OWA_mas * 1.1, max_sep * 1.1))
    ax1.legend(loc='best', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25, ls=':')

    # ---- Phase Angle ----
    ax2 = axes[ax_idx]; ax_idx += 1
    ax2.fill_between(years, csv_data['phase_angle_deg_2.5th'],
                     csv_data['phase_angle_deg_97.5th'],
                     color=c_fill_95, alpha=0.3, label='95% CI')
    ax2.fill_between(years, csv_data['phase_angle_deg_16th'],
                     csv_data['phase_angle_deg_84th'],
                     color=c_fill_68, alpha=0.5, label='68% CI')
    ax2.plot(years, csv_data['phase_angle_deg_median'], '-',
             color=c_median, lw=2.5, label='Median', marker='o', markersize=2)
    ax2.set_ylabel('Phase Angle (deg)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 180)
    ax2.legend(loc='best', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.25, ls=':')

    # ---- Flux Contrast ----
    ax3 = axes[ax_idx]; ax_idx += 1
    if 'flux_contrast_median' in csv_data.columns:
        ax3.fill_between(years, csv_data['flux_contrast_2.5th'],
                         csv_data['flux_contrast_97.5th'],
                         color=c_fill_95, alpha=0.3, label='95% CI')
        ax3.fill_between(years, csv_data['flux_contrast_16th'],
                         csv_data['flux_contrast_84th'],
                         color=c_fill_68, alpha=0.5, label='68% CI')
        ax3.plot(years, csv_data['flux_contrast_median'], '-',
                 color=c_median, lw=2.5, label='Median',
                 marker='o', markersize=2)
        ax3.set_yscale('log')
        ax3.set_ylabel('Flux Contrast', fontsize=11, fontweight='bold')
        ax3.set_ylim(1e-11, 1e-7)
        ax3.legend(loc='best', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.25, ls=':')

    # ---- Integration Time ----
    if has_inttime:
        ax_int = axes[ax_idx]; ax_idx += 1
        for lbl, prefix, c in [
            ('Optimistic', 'integration_time_hours_opt', c_opt),
            ('Conservative', 'integration_time_hours_con', c_con),
        ]:
            valid = np.isfinite(csv_data[f'{prefix}_median'].values)
            if np.any(valid):
                y_med = csv_data[f'{prefix}_median'].values.copy()
                y16 = csv_data[f'{prefix}_16th'].values.copy()
                y84 = csv_data[f'{prefix}_84th'].values.copy()
                y025 = csv_data[f'{prefix}_2.5th'].values.copy()
                y975 = csv_data[f'{prefix}_97.5th'].values.copy()
                for arr in (y_med, y16, y84, y025, y975):
                    arr[~valid] = np.nan
                ax_int.fill_between(years, y025, y975, color=c, alpha=0.15,
                                    label=f'{lbl} 95% CI')
                ax_int.fill_between(years, y16, y84, color=c, alpha=0.30,
                                    label=f'{lbl} 68% CI')
                ax_int.plot(years, y_med, '-', color=c, lw=2.5,
                            label=f'{lbl} median', marker='o', markersize=2)
        ax_int.set_yscale('log')
        ax_int.set_ylabel('Integration Time (hours)',
                          fontsize=11, fontweight='bold')
        ax_int.legend(loc='best', fontsize=8, framealpha=0.9)
        ax_int.grid(True, alpha=0.25, ls=':')

    # ---- Observation windows ----
    for i, ax in enumerate(axes):
        ylims = ax.get_ylim()
        if targ_observable is not None:
            ax.fill_between(
                years, ylims[0], ylims[1], where=~targ_observable,
                alpha=0.12, color='gray',
                label='Solar keepout' if i == 0 else '', zorder=0)
        if GB_not_observable is not None:
            ax.fill_between(
                years, ylims[0], ylims[1], where=~GB_not_observable,
                alpha=0.12, color='orange',
                label='GB obs' if i == 0 else '', zorder=0)
        ax.set_ylim(ylims)
        ax.set_xlim(years[0], years[-1])
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel('Year', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plot_fpath = os.path.join(
        output_dir,
        f'{fname_prefix}_Band{band}_inttime_contrast_comparison.{fig_ext}')
    plt.savefig(plot_fpath, dpi=150, bbox_inches='tight')
    print(f'  Plot saved to {plot_fpath}')
    if show_plots:
        plt.show()
    plt.close('all')