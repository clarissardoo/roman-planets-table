import numpy as np
import pandas as pd
from astropy.time import Time
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit
from astropy import units as u
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os, pickle, warnings


# Display names for prettier output
display_names={
    "47_UMa":"47 UMa c",
    "55_Cnc":"55 Cancri d",
    "eps_Eri":"Eps Eri b",
    "HD_87883":"HD 87883 b",
    "HD_114783":"HD 114783 c",
    "HD_134987":"HD 134987 c",
    "HD_154345":"HD 154345 b",
    "HD_160691":"HD 160691 e",
    "HD_190360":"HD 190360 b",
    "HD_217107":"HD 217107 c",
    "pi_Men":"Pi Men b",
    "ups_And":"Ups And d",
    "HD_192310":"HD 192310 c",
}


orbit_params={
    "47_UMa":{
        "basis":"per tc secosw sesinw k",
        "m0":1.0051917028549999,"m0_err":0.0468882076437500,
        "plx":72.452800,"plx_err":0.150701,
        "n_planets":3,"pl_num":2,"g_mag":4.866588,
    },
    "55_Cnc":{
        "basis":"per tc secosw sesinw k",
        "m0":0.905,"m0_err":0.015,
        "plx":79.4274000,"plx_err":0.0776646,
        "n_planets":5,"pl_num":3,"g_mag":5.732681,
    },
    "eps_Eri":{
        "basis":"per tc secosw sesinw k",
        "m0":0.82,"m0_err":0.02,
        "plx":312.219000,"plx_err":0.467348,
        "n_planets":1,"pl_num":1,"g_mag":3.465752,"inc_mean":78.810,"inc_sig":29.340
    },
    "HD_87883":{
        "basis":"per tc secosw sesinw k",
        "m0":0.810,"m0_err":0.091,
        "plx":54.6421000,"plx_err":0.0369056,
        "n_planets":1,"pl_num":1,"g_mag":7.286231,"inc_mean":25.45,"inc_sig":1.61
    },
    "HD_114783":{
        "basis":"per tc secosw sesinw k",
        "m0":0.90,"m0_err":0.04,
        "plx":47.4482000,"plx_err":0.0637202,
        "n_planets":2,"pl_num":2,"g_mag":7.330857,"inc_mean":159,"inc_sig":6
    },
    "HD_134987":{
        "basis":"per tc secosw sesinw k",
        "m0":1.0926444945650000,"m0_err":0.0474835459017250,
        "plx":38.1678000,"plx_err":0.0746519,
        "n_planets":2,"pl_num":2,"g_mag":6.302472,
    },
    "HD_154345":{
        "basis":"per tc secosw sesinw k",
        "m0":0.88,"m0_err":0.09,
        "plx":54.6636000,"plx_err":0.0212277,
        "n_planets":1,"pl_num":1,"g_mag":6.583667,"inc_mean":69,"inc_sig":13
    },
    "HD_160691":{
        "basis":"per tc secosw sesinw k",
        "m0":1.13,"m0_err":0.02,
        "plx":64.082,"plx_err":0.120162,
        "n_planets":4,"pl_num":4,"g_mag":4.942752,
    },
    "HD_190360":{
        "basis":"per tc secosw sesinw k",
        "m0":1.0,"m0_err":0.1,
        "plx":62.4443000,"plx_err":0.0616881,
        "n_planets":2,"pl_num":1,"g_mag":5.552787,"inc_mean":80.2,"inc_sig":23.2
    },
    "HD_217107":{
        "basis":"per tc secosw sesinw k",
        "m0":1.05963082882500,"m0_err":0.04470613802572,
        "plx":49.8170000,"plx_err":0.0573616,
        "n_planets":2,"pl_num":2,"g_mag":5.996743,"inc_mean":89.3,"inc_sig":9.0
    },
    "pi_Men":{
        "basis":"per tc secosw sesinw k",
        "m0":1.10,"m0_err":0.14,
        "plx":54.705200,"plx_err":0.067131,
        "n_planets":1,"pl_num":1,"g_mag":5.511580,"inc_mean":54.436,"inc_sig":5.945
    },
    "ups_And":{
        "basis":"per tc secosw sesinw k",
        "m0":1.29419667430000,"m0_err":0.04122482369025,
        "plx":74.571100,"plx_err":0.349118,
        "n_planets":3,"pl_num":3,"g_mag":3.966133,"inc_mean":23.758,"inc_sig":1.316
    },
    "HD_192310":{
        "basis":"per tc secosw sesinw k",
        "m0":0.84432448757250,"m0_err":0.02820926681885,
        "plx":113.648000,"plx_err":0.118606,
        "n_planets":2,"pl_num":2,"g_mag":5.481350
    },
}


def compute_sep(
        df,epochs,basis,m0,m0_err,plx,plx_err,n_planets=1,pl_num=1,
        override_inc=None,override_lan=None,inc_mean=None,inc_sig=None,
        user_inc_mean=None,user_inc_sig=None
):
    """
    Computes a sky-projected angular separation posterior given a
    RadVel-computed DataFrame.

    Args:
        df (pd.DataFrame): Radvel-computed posterior (in any orbital basis)
        epochs (np.array of astropy.time.Time): epochs at which to compute
            separations
        basis (str): basis string of input posterior (see
            radvel.basis.BASIS_NAMES` for the full list of possibilities).
        m0 (float): median of primary mass distribution (assumed Gaussian).
        m0_err (float): 1sigma error of primary mass distribution
            (assumed Gaussian).
        plx (float): median of parallax distribution (assumed Gaussian).
        plx_err: 1sigma error of parallax distribution (assumed Gaussian).
        n_planets (int): total number of planets in RadVel posterior
        pl_num (int): planet number used in RadVel fits (e.g. a RadVel label of
            'per1' implies `pl_num` == 1)
        override_inc (float or str): Fixed inclination in degrees, or None/"gaussian" for sampling
        inc_mean (float): Mean inclination in degrees for Gaussian sampling (from orbit_params)
        inc_sig (float): Standard deviation of inclination in degrees for Gaussian sampling (from orbit_params)
        user_inc_mean (float): User-provided mean inclination in degrees
        user_inc_sig (float): User-provided standard deviation of inclination in degrees

    Returns:
        tuple of:
            np.array of size (len(epochs) x len(df)): sky-projected angular
                separations [mas] at each input epoch
            np.array: RA offsets [mas]
            np.array: Dec offsets [mas]
            np.array: planet masses in solar masses
            np.array: inclinations in radians
            np.array: true anomaly
            np.array: z component [mas]
            np.array: 3D orbital radius [AU]
            np.array: 3D orbital radius [mas]
    """

    myBasis=Basis(basis,n_planets)
    df=myBasis.to_synth(df)
    chain_len=len(df)
    tau_ref_epoch=58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st=np.random.normal(m0,m0_err,size=chain_len)
    semiamp=df['k{}'.format(pl_num)].values
    per_day=df['per{}'.format(pl_num)].values
    period_yr=per_day/365.25
    ecc=df['e{}'.format(pl_num)].values
    msini=(
            Msini(semiamp,per_day,m_st,ecc,Msini_units='Earth')*
            (u.M_earth/u.M_sun).to('')
    )

    #median msini for critical inclination
    median_msini=np.median(msini)

    # Handle inclination sampling
    if user_inc_mean is not None and user_inc_sig is not None:
        # User-provided Gaussian distribution
        inc_deg_samples=np.random.normal(user_inc_mean,user_inc_sig,size=chain_len)
        # Clip to valid range [0, 180] degrees
        inc_deg_samples=np.clip(inc_deg_samples,0,180)
        inc=np.radians(inc_deg_samples)
    elif override_inc is not None and override_inc!="gaussian":
        # Fixed inclination provided by user
        inc=np.full(chain_len,np.radians(override_inc))
    elif override_inc=="gaussian" and inc_mean is not None and inc_sig is not None:
        # Sample from Gaussian distribution (from orbit_params)
        inc_deg_samples=np.random.normal(inc_mean,inc_sig,size=chain_len)
        # Clip to valid range [0, 180] degrees
        inc_deg_samples=np.clip(inc_deg_samples,0,180)
        inc=np.radians(inc_deg_samples)
    else:
        # Default: uniform in cos(inc) random sampling with critical inclination constraint
        # Critical inclination: where planet becomes a star (0.08 Msun)
        crit_incrad=np.arcsin(median_msini/0.08)
        cosi=(2.*np.random.random(size=chain_len)*np.cos(crit_incrad))-1.
        inc=np.arccos(cosi)

    m_pl=msini/np.sin(inc)
    mtot=m_st+m_pl
    sma=(period_yr**2*mtot)**(1/3)
    omega_st_rad=df['w{}'.format(pl_num)].values
    omega_pl_rad=omega_st_rad+np.pi
    parallax=np.random.normal(plx,plx_err,size=chain_len)

    if override_lan is not None:
        lan=np.full(chain_len,np.radians(override_lan))
    else:
        lan=np.random.random_sample(size=chain_len)*2.*np.pi

    tp_mjd=df['tp{}'.format(pl_num)].values-2400000.5
    tau=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)

    # compute projected separation in mas
    raoff,deoff,vz=calc_orbit(
        epochs.mjd,sma,ecc,inc,
        omega_pl_rad,lan,tau,
        parallax,mtot,tau_ref_epoch=tau_ref_epoch
    )
    seps=np.sqrt(raoff**2+deoff**2)

    # Compute true anomaly and 3D positions for each epoch
    n_epochs=len(epochs)
    true_anomaly=np.zeros((n_epochs,chain_len))
    x_mas=np.zeros((n_epochs,chain_len))
    y_mas=np.zeros((n_epochs,chain_len))
    z_mas=np.zeros((n_epochs,chain_len))

    # Thiele-Innes from my orbit_getpoints
    A=sma*(np.cos(omega_pl_rad)*np.cos(lan)-np.sin(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    B=sma*(np.cos(omega_pl_rad)*np.sin(lan)+np.sin(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    F=sma*(-np.sin(omega_pl_rad)*np.cos(lan)-np.cos(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    G=sma*(-np.sin(omega_pl_rad)*np.sin(lan)+np.cos(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    C=sma*np.sin(omega_pl_rad)*np.sin(inc)
    H=sma*np.cos(omega_pl_rad)*np.sin(inc)

    for i in range(n_epochs):
        # Mean anom
        n_motion=2*np.pi/per_day  # mean motion (rad/day)
        M=n_motion*(epochs.mjd[i]-tp_mjd)

        # eccentric anomaly from orbit_getpoints
        EA=M+ecc*np.sin(M)+ecc**2*np.sin(2*M)/2
        for _ in range(20):
            err=EA-ecc*np.sin(EA)-M
            if np.all(np.abs(err)<1e-15):
                break
            EA=EA-err/(1-ecc*np.cos(EA))

        # True anomaly
        f=2*np.arctan2(
            np.sqrt(1+ecc)*np.sin(EA/2),
            np.sqrt(1-ecc)*np.cos(EA/2)
        )
        true_anomaly[i,:]=f

        # Position in orbital plane
        X=np.cos(EA)-ecc
        Y=np.sqrt(1-ecc**2)*np.sin(EA)

        # 3D position in AU
        X_au=(B*X+G*Y)
        Y_au=(A*X+F*Y)
        Z_au=(C*X+H*Y)

        # Convert to mas
        x_mas[i,:]=X_au*parallax
        y_mas[i,:]=Y_au*parallax
        z_mas[i,:]=Z_au*parallax

    # 3D orbital radius
    r_au=np.sqrt(x_mas**2+y_mas**2+z_mas**2)/parallax  #AU
    z_au=z_mas/parallax  #AU

    return seps,raoff,deoff,m_pl,inc,true_anomaly,z_au,r_au,parallax


def weighted_percentile(data,weights,percentile):
    """Compute weighted percentile given posteriors and ln-like weights from posteriors sampled"""
    result=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        sorted_indices=np.argsort(data[i,:])
        sorted_data=data[i,sorted_indices]
        sorted_weights=weights[sorted_indices]
        cumsum=np.cumsum(sorted_weights)
        cutoff=percentile/100.0
        idx=np.searchsorted(cumsum,cutoff)
        if idx>=len(sorted_data):
            idx=len(sorted_data)-1
        result[i]=sorted_data[idx]
    return result


def weighted_mean(data,weights):
    """Compute weighted mean along axis 1"""
    if data.ndim==1:
        return np.average(data,weights=weights)
    else:
        return np.average(data,axis=1,weights=weights)


def weighted_std(data,weights):
    """Compute weighted standard deviation along axis 1"""
    if data.ndim==1:
        mean=np.average(data,weights=weights)
        variance=np.average((data-mean)**2,weights=weights)
        return np.sqrt(variance)
    else:
        mean=np.average(data,axis=1,weights=weights)
        variance=np.average((data-mean[:,np.newaxis])**2,axis=1,weights=weights)
        return np.sqrt(variance)


def parse_inclination(inc_str):
    """Parse inclination input string and return (mode, value, uncertainty)"""
    inc_str=inc_str.strip().lower()

    if inc_str=='random':
        return ('random',None,None)
    elif inc_str=='gaussian':
        return ('gaussian',None,None)

    import re

    # Check for ± symbol
    match=re.match(r'^(\d+\.?\d*)\s*±\s*(\d+\.?\d*)$',inc_str)
    if match:
        mean_val=float(match.group(1))
        sigma_val=float(match.group(2))

        if mean_val<0 or mean_val>180:
            raise ValueError(f"Inclination mean must be between 0 and 180 degrees (got {mean_val})")

        if sigma_val<=0:
            raise ValueError(f"Inclination uncertainty must be positive (got {sigma_val})")

        lower_3sigma=mean_val-3*sigma_val
        upper_3sigma=mean_val+3*sigma_val

        if lower_3sigma<-90 or upper_3sigma>270:
            raise ValueError(
                f"Inclination uncertainty too large (μ={mean_val}°, σ={sigma_val}°). "
                f"The 3σ range [{lower_3sigma:.1f}°, {upper_3sigma:.1f}°] extends too far outside [0°, 180°]. "
                f"Consider using a smaller uncertainty."
            )

        return ('user_gaussian',mean_val,sigma_val)

    # +/- notation
    match=re.match(r'^(\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)$',inc_str)
    if match:
        mean_val=float(match.group(1))
        sigma_val=float(match.group(2))

        if mean_val<0 or mean_val>180:
            raise ValueError(f"Inclination mean must be between 0 and 180 degrees (got {mean_val})")

        if sigma_val<=0:
            raise ValueError(f"Inclination uncertainty must be positive (got {sigma_val})")

        lower_3sigma=mean_val-3*sigma_val
        upper_3sigma=mean_val+3*sigma_val

        if lower_3sigma<-90 or upper_3sigma>270:
            raise ValueError(
                f"Inclination uncertainty too large (μ={mean_val}°, σ={sigma_val}°). "
                f"The 3σ range [{lower_3sigma:.1f}°, {upper_3sigma:.1f}°] extends too far outside [0°, 180°]. "
                f"Consider using a smaller uncertainty."
            )

        return ('user_gaussian',mean_val,sigma_val)

    # Plain number
    try:
        value=float(inc_str)

        if value<0 or value>180:
            raise ValueError(f"Inclination must be between 0 and 180 degrees (got {value})")

        return ('fixed',value,None)
    except ValueError as e:
        if "Inclination must be between" in str(e):
            raise
        raise ValueError(f"Invalid inclination format: '{inc_str}'")


def compute_orbit_for_plotting(df,epochs,basis,m0,m0_err,plx,plx_err,
                               n_planets=1,pl_num=1,override_inc=None,
                               override_lan=None,inc_mean=None,inc_sig=None,
                               user_inc_mean=None,user_inc_sig=None):
    """
    Compute orbit trajectories for 2D plotting (RA/Dec offsets).
    Returns the same separation data as compute_sep for plots.
    """
    myBasis=Basis(basis,n_planets)
    df=myBasis.to_synth(df)
    chain_len=len(df)
    tau_ref_epoch=58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st=np.random.normal(m0,m0_err,size=chain_len)
    semiamp=df['k{}'.format(pl_num)].values
    per_day=df['per{}'.format(pl_num)].values
    period_yr=per_day/365.25
    ecc=df['e{}'.format(pl_num)].values
    msini=(
            Msini(semiamp,per_day,m_st,ecc,Msini_units='Earth')*
            (u.M_earth/u.M_sun).to('')
    )

    # Calculate median msini for critical inclination
    median_msini=np.median(msini)

    # Handle inc sampling
    if user_inc_mean is not None and user_inc_sig is not None:
        inc_deg_samples=np.random.normal(user_inc_mean,user_inc_sig,size=chain_len)
        inc_deg_samples=np.clip(inc_deg_samples,0,180)
        inc=np.radians(inc_deg_samples)
    elif override_inc is not None and override_inc!="gaussian":
        inc=np.full(chain_len,np.radians(override_inc))
    elif override_inc=="gaussian" and inc_mean is not None and inc_sig is not None:
        inc_deg_samples=np.random.normal(inc_mean,inc_sig,size=chain_len)
        inc_deg_samples=np.clip(inc_deg_samples,0,180)
        inc=np.radians(inc_deg_samples)
    else:
        # Default: uniform random sampling with critical inclination constraint
        crit_incrad=np.arcsin(median_msini/0.08)
        cosi=(2.*np.random.random(size=chain_len)*np.cos(crit_incrad))-1.
        inc=np.arccos(cosi)

    m_pl=msini/np.sin(inc)
    mtot=m_st+m_pl
    sma=(period_yr**2*mtot)**(1/3)
    omega_st_rad=df['w{}'.format(pl_num)].values
    omega_pl_rad=omega_st_rad+np.pi
    parallax=np.random.normal(plx,plx_err,size=chain_len)

    if override_lan is not None:
        lan=np.full(chain_len,np.radians(override_lan))
    else:
        lan=np.random.random_sample(size=chain_len)*2.*np.pi

    tp_mjd=df['tp{}'.format(pl_num)].values-2400000.5
    tau=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)

    # compute projected separation in mas
    raoff,deoff,vz=calc_orbit(
        epochs.mjd,sma,ecc,inc,
        omega_pl_rad,lan,tau,
        parallax,mtot,tau_ref_epoch=tau_ref_epoch
    )

    # Get best-fit index (I don't use this anymore for plots)
    lnlike=df["lnprobability"].values
    best_idx=np.argmax(lnlike)

    return raoff,deoff,best_idx


def gen_point_cloud(planet,
                  params=None, #override default planet params
                  posterior_dir='orbit_fits',
                  output_dir='.',
                  start_date='2027-01-01',
                  end_date='2027-06-01',
                  time_interval=1,
                  inc_mode='random',
                  inc_params=None,
                  override_lan=0.,
                  nsamp='all',
                  out_fname=None):
    
    base_path=Path(posterior_dir)
    planet_dir=base_path/planet
    files=list(planet_dir.glob("*.csv.bz2"))
    if not files:
        raise UserWarning(f"Error: No posterior data found for {planet} in {planet_dir}")
    if len(files)>1:
        raise UserWarning(f"Multiple posterior data files found for {planet} in {planet_dir}")
    print(f"Loading posterior data from {files[0]}...")
    df=pd.read_csv(files[0])
    if nsamp=='all':
        nsamp=len(df)
        print(f"Using all {nsamp} posterior samples")

    print()
    print("-"*60)
    print(f"Configuration:")
    print(f"  Planet: {display_names[planet]}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Time interval: {time_interval} days")

    if params is None:
        params = orbit_params[planet]

    if inc_mode=='user_gaussian':
        inc_value,inc_uncertainty=inc_params
        print(f"  Inclination: Gaussian (μ={inc_value:.1f}°, σ={inc_uncertainty:.1f}°) [user-defined]")
        inc_display=f"{inc_value:.1f}±{inc_uncertainty:.1f}"
        override_inc=None
        user_inc_mean=inc_value
        user_inc_sig=inc_uncertainty
    elif inc_mode=='gaussian':
        has_gaussian_info="inc_mean" in params and "inc_sig" in params
        if has_gaussian_info:
            print(f"  Inclination: Gaussian (μ={params['inc_mean']:.1f}°, σ={params['inc_sig']:.1f}°)")
            inc_display=f"gaussian (μ={params['inc_mean']:.1f}°, σ={params['inc_sig']:.1f}°)"
            override_inc="gaussian"
            user_inc_mean=None
            user_inc_sig=None
        else:
            print(f"  No gaussian priors available. Falling back to random inclination.")
            inc_mode="random"
            inc_display="random"
    elif inc_mode=='fixed':
        inc_value=inc_params[0]
        print(f"  Inclination: {inc_value:.1f}° (fixed)")
        inc_display=f"{inc_value:.1f}"
        override_inc=inc_value
        user_inc_mean=None
        user_inc_sig=None
    else:
        print(f"  Inclination: random (uniform with critical inclination constraint)")
        inc_mode="random"
        inc_display="random"
        override_inc=None
        user_inc_mean=None
        user_inc_sig=None

    print(f"  Posterior samples: {nsamp}")
    print("-"*60)
    print()

    try:
        t_start=Time(start_date)
        t_end=Time(end_date)
    except:
        raise UserWarning("Error: Invalid date format. Use YYYY-MM-DD (e.g., 2026-06-01)")

    if t_end<=t_start:
        raise ValueError("Error: End date must be after start date")

    print(f"Sampling {nsamp} orbits from posterior...")
    df_sample=df.sample(nsamp,replace=True) # TODO: Consider if replace=True is the right choice statistically? -Ell

    n_epochs=int((t_end.mjd-t_start.mjd)/time_interval)+1
    epochs=Time(np.linspace(t_start.mjd,t_end.mjd,n_epochs),format="mjd")

    print(f"Generating point cloud for {n_epochs} epochs...")

    # output shape of each array is (n_epochs,nsamp)
    seps_mas,raoff_mas,deoff_mas,m_pl,inc,true_anomaly,z_au,r_au,parallax=compute_sep(
        df_sample,epochs,
        params["basis"],params["m0"],params["m0_err"],
        params["plx"],params["plx_err"],
        params["n_planets"],params["pl_num"],
        override_lan=override_lan,
        override_inc=override_inc,
        inc_mean=params.get("inc_mean"),
        inc_sig=params.get("inc_sig"),
        user_inc_mean=user_inc_mean,
        user_inc_sig=user_inc_sig
    )

    # Phase angle calculation using 3D radius
    phase_angle_rad=np.arccos(z_au/r_au)
    phase_angle_deg=np.degrees(phase_angle_rad)

    lambert_phase=(np.sin(phase_angle_rad)+(np.pi-phase_angle_rad)*np.cos(phase_angle_rad))/np.pi

    m_pl_mjup=m_pl*(u.M_sun/u.M_jup).to('')
    m_pl_mearth=m_pl*(u.M_sun/u.M_earth).to('')

    mass_intervals=np.array([0,2.04,95.16,317.828407,26635.6863,np.inf])
    C=np.array([0.00346053,-0.06613329,0.48091861,1.04956612,-2.84926757])
    S=np.array([0.279,0.50376436,0.22725968,0,0.881])
    r_pl_rearth=np.zeros_like(m_pl_mearth)
    for i in range(len(mass_intervals)-1):
        mask=(m_pl_mearth>=mass_intervals[i])&(m_pl_mearth<mass_intervals[i+1])
        if np.any(mask):
            r_pl_rearth[mask]=10**(C[i]+S[i]*np.log10(m_pl_mearth[mask]))

    r_pl_rjup=r_pl_rearth*(u.R_earth/u.R_jup).to('')

    inc_deg=np.degrees(inc)

    # Get lnlike for weighting the posteriors
    myBasis=Basis(params["basis"],params["n_planets"])
    df_synth=myBasis.to_synth(df_sample)
    lnlike=df_synth["lnprobability"].values    

    
    # Organize output cloud
    lnlike_arr = np.full_like(seps_mas,lnlike) # ln likelihood
    epochs_mjd_arr = np.full_like(seps_mas,epochs.value[:,np.newaxis])
    distance_pc = np.full_like(seps_mas,1000.0/parallax)
    mass_arr = np.full_like(seps_mas,m_pl_mjup)
    pl_rad_arr = np.full_like(seps_mas,r_pl_rjup)
    incdeg_arr = np.full_like(seps_mas,inc_deg)

    point_cloud = {
        'epoch_mjd' : epochs_mjd_arr,
        'sep_mas' : seps_mas,
        'raoff_mas' : raoff_mas,
        'deoff_mas' : deoff_mas,
        'true_anom_deg' : true_anomaly,
        'z_au' : z_au,
        'orbital_radius_au' : r_au,
        'phase_angle_deg' : phase_angle_deg,
        'lambert_phase' : lambert_phase,
        'm_pl_mjup' : mass_arr,
        'r_pl_rjup' : pl_rad_arr,
        'inc_deg' : incdeg_arr,
        'ln_likelihood' : lnlike_arr,
        'dist_pc' : distance_pc
    }

    # Save pickle
    if out_fname is None:
        planet_name=planet.replace("_","")
        output_file=f"{planet_name}_{start_date}_to_{end_date}_PointCloud.pkl"
    else:
        output_file = out_fname.split('.')[0] + '_PointCloud.pkl'
    output_fpath=os.path.join(output_dir,output_file)    
    print(f'Saving point cloud to {output_fpath}')
    with open(output_fpath,'wb') as f:
        pickle.dump(point_cloud,f)

    return point_cloud


def gen_summary_csv(planet,params,
                  posterior_dir='orbit_fits',
                  output_dir='.',
                  start_date='2027-01-01',
                  end_date='2027-06-01',
                  time_interval=1,
                  inc_mode='random',
                  inc_params=None,
                  override_lan=0.,
                  nsamp='all',
                  output=None,
                  plot=True,
                  show_plots=False):

    mass_median=np.median(m_pl_mjup)
    mass_16th=np.percentile(m_pl_mjup,16)
    mass_84th=np.percentile(m_pl_mjup,84)
    mass_err_lower=mass_median-mass_16th
    mass_err_upper=mass_84th-mass_median
    rad_median=np.median(r_pl_rjup)
    rad_16th=np.percentile(r_pl_rjup,16)
    rad_84th=np.percentile(r_pl_rjup,84)
    rad_err_lower=rad_median-rad_16th
    rad_err_upper=rad_84th-rad_median
    inc_median=np.median(inc_deg)
    inc_16th=np.percentile(inc_deg,16)
    inc_84th=np.percentile(inc_deg,84)

    print(f"Planet mass: {mass_median:.2f} +{mass_err_upper:.2f}/-{mass_err_lower:.2f} M_Jup")
    print(f"Planet radius: {rad_median:.2f} +{rad_err_upper:.2f}/-{rad_err_lower:.2f} R_Jup")
    print(f"Inclination: {inc_median:.2f} [{inc_16th:.2f}, {inc_84th:.2f}] degrees")
    print()

    # This is where we weight the posteriors by lnlike



    weights=np.exp(lnlike-np.max(lnlike))
    weights=weights/np.sum(weights)

    # Compute percentiles, mean, and std for all quantities
    med_sep=weighted_percentile(seps,weights,50)
    low_sep=weighted_percentile(seps,weights,16)
    high_sep=weighted_percentile(seps,weights,84)
    low_sep_95=weighted_percentile(seps,weights,2.5)
    high_sep_95=weighted_percentile(seps,weights,97.5)
    mean_sep=weighted_mean(seps,weights)
    std_sep=weighted_std(seps,weights)

    # med_rad_au=med_sep*distance_pc/1000.0
    # low_rad_au=low_sep*distance_pc/1000.0
    # high_rad_au=high_sep*distance_pc/1000.0
    # low_rad_au_95=low_sep_95*distance_pc/1000.0
    # high_rad_au_95=high_sep_95*distance_pc/1000.0
    # mean_rad_au=mean_sep*distance_pc/1000.0
    # std_rad_au=std_sep*distance_pc/1000.0

    # 3D orbital radius statistics
    med_r_au=weighted_percentile(r_au,weights,50)
    low_r_au=weighted_percentile(r_au,weights,16)
    high_r_au=weighted_percentile(r_au,weights,84)
    low_r_au_95=weighted_percentile(r_au,weights,2.5)
    high_r_au_95=weighted_percentile(r_au,weights,97.5)
    mean_r_au=weighted_mean(r_au,weights)
    std_r_au=weighted_std(r_au,weights)

    med_phase=weighted_percentile(phase_angle_deg,weights,50)
    low_phase=weighted_percentile(phase_angle_deg,weights,16)
    high_phase=weighted_percentile(phase_angle_deg,weights,84)
    low_phase_95=weighted_percentile(phase_angle_deg,weights,2.5)
    high_phase_95=weighted_percentile(phase_angle_deg,weights,97.5)
    mean_phase=weighted_mean(phase_angle_deg,weights)
    std_phase=weighted_std(phase_angle_deg,weights)

    med_lambert_phase=weighted_percentile(lambert_phase,weights,50)
    low_lambert_phase=weighted_percentile(lambert_phase,weights,16)
    high_lambert_phase=weighted_percentile(lambert_phase,weights,84)
    low_lambert_phase_95=weighted_percentile(lambert_phase,weights,2.5)
    high_lambert_phase_95=weighted_percentile(lambert_phase,weights,97.5)
    mean_lambert_phase=weighted_mean(lambert_phase,weights)
    std_lambert_phase=weighted_std(lambert_phase,weights)

    true_anomaly_deg=np.degrees(true_anomaly)
    true_anomaly_deg=true_anomaly_deg%360
    med_nu=weighted_percentile(true_anomaly_deg,weights,50)
    low_nu=weighted_percentile(true_anomaly_deg,weights,16)
    high_nu=weighted_percentile(true_anomaly_deg,weights,84)
    low_nu_95=weighted_percentile(true_anomaly_deg,weights,2.5)
    high_nu_95=weighted_percentile(true_anomaly_deg,weights,97.5)
    mean_nu=weighted_mean(true_anomaly_deg,weights)
    std_nu=weighted_std(true_anomaly_deg,weights)

    csv_data=pd.DataFrame({
        'date_iso':epochs.iso,
        'mjd':epochs.mjd,

        'decimal_year':epochs.decimalyear,
        'separation_mas_median':med_sep,
        'separation_mas_16th':low_sep,
        'separation_mas_84th':high_sep,
        'separation_mas_2.5th':low_sep_95,
        'separation_mas_97.5th':high_sep_95,
        'separation_mas_mean':mean_sep,
        'separation_mas_std':std_sep,

        'orbital_radius_au_median':med_r_au,
        'orbital_radius_au_16th':low_r_au,
        'orbital_radius_au_84th':high_r_au,
        'orbital_radius_au_2.5th':low_r_au_95,
        'orbital_radius_au_97.5th':high_r_au_95,
        'orbital_radius_au_mean':mean_r_au,
        'orbital_radius_au_std':std_r_au,

        'phase_angle_deg_median':med_phase,
        'phase_angle_deg_16th':low_phase,
        'phase_angle_deg_84th':high_phase,
        'phase_angle_deg_2.5th':low_phase_95,
        'phase_angle_deg_97.5th':high_phase_95,
        'phase_angle_deg_mean':mean_phase,
        'phase_angle_deg_std':std_phase,

        'lambert_phase_median':med_lambert_phase,
        'lambert_phase_16th':low_lambert_phase,
        'lambert_phase_84th':high_lambert_phase,
        'lambert_phase_2.5th':low_lambert_phase_95,
        'lambert_phase_97.5th':high_lambert_phase_95,
        'lambert_phase_mean':mean_lambert_phase,
        'lambert_phase_std':std_lambert_phase,

        'true_anomaly_deg_median':med_nu,
        'true_anomaly_deg_16th':low_nu,
        'true_anomaly_deg_84th':high_nu,
        'true_anomaly_deg_2.5th':low_nu,
        'true_anomaly_deg_97.5th':high_nu,
        'true_anomaly_deg_mean':mean_nu,
        'true_anomaly_deg_std':std_nu,
    })

    # output file name
    if output is None:
        planet_name=planet.replace("_","")
        output_file=f"{planet_name}_separations_{start_date}_to_{end_date}.csv"
    else:
        output_file=output

    output_fpath=os.path.join(output_dir,output_file)
    print(f"Writing output to {output_fpath}...")
    with open(output_fpath,'w') as f:
        f.write(f"# Planet: {display_names[planet]}\n")
        f.write(f"# Date range: {start_date} to {end_date}\n")
        f.write(f"# Time interval: {time_interval} days\n")
        f.write(f"# Inclination: {inc_display}\n")
        f.write(f"# Number of posterior samples: {nsamp}\n")
        f.write(f"# Number of epochs: {n_epochs}\n")
        f.write(f"#\n")
        f.write(f"# System parameters:\n")
        f.write(f"# Distance: {distance_pc:.2f} pc (parallax: {params['plx']:.2f} +/- {params['plx_err']:.2f} mas)\n")
        f.write(f"#\n")
        f.write(f"# Derived parameters:\n")
        f.write(f"# Planet mass: {mass_median:.3f} +{mass_err_upper:.3f}/-{mass_err_lower:.3f} M_Jup\n")
        f.write(
            f"# Planet radius: {rad_median:.3f} +{rad_err_upper:.3f}/-{rad_err_lower:.3f} R_Jup\n")
        f.write(
            f"# Inclination distribution: {inc_median:.2f} deg (median), [{inc_16th:.2f}, {inc_84th:.2f}] deg (16th-84th percentile)\n")
        f.write("#\n")
        csv_data.to_csv(f,index=False)

    print(f"Output saved to {output_fpath}")
    print(f"\nSummary:")
    print(f"  Planet: {display_names[planet]}")
    print(f"  Distance: {distance_pc:.2f} pc")
    print(f"  Epochs: {n_epochs}")
    print(
        f"  Separation range: {med_sep.min():.2f} - {med_sep.max():.2f} mas")

    # Generate plots if requested
    if plot:
        print("\nGenerating plots...")
        output_prefix=output_fpath.replace('.csv','')
        plot_orbital_parameters(
            csv_data,
            display_names[planet],
            output_prefix,
            df_sample=df_sample,
            params=params,
            # override_inc=override_inc,
            override_lan=override_lan,
            # user_inc_mean=user_inc_mean,
            # user_inc_sig=user_inc_sig,
            start_date=start_date,
            end_date=end_date,
            fig_ext='pdf',
            show_plots=show_plots
        )

    return df_sample,csv_data


def plot_orbital_parameters(csv_data,planet_name,output_prefix,
                            df_sample=None,params=None,override_inc=None,
                            override_lan=None,user_inc_mean=None,user_inc_sig=None,
                            start_date=None,end_date=None,figsize=None,fig_ext='png',
                            show_plots=False):
    """
    Create plots including 2D orbits and time-series parameters.

    Args:
        csv_data (pd.DataFrame): DataFrame with computed orbital parameters
        planet_name (str): Display name of the planet
        output_prefix (str): Prefix for output plot files
        df_sample (pd.DataFrame): Sampled posterior data (for orbit plots)
        params (dict): Orbital parameters dictionary (for orbit plots)
        override_inc (float): Fixed inclination override
        override_lan (float): Fixed longitude of ascending node override
        user_inc_mean (float): User-provided inclination mean
        user_inc_sig (float): User-provided inclination std dev
        start_date (str): Start date for orbit plot
        end_date (str): End date for orbit plot
        figsize (tuple of float): Figure size in inches
        fig_ext (str): file extension for saved figure, defaults to 'png'
        show_plots (bool): display the figure in output stream, defaults False
    """
    # Convert decimal years for plotting
    years=csv_data['decimal_year'].values

    # Plasma colormap colors
    cm=plt.cm.plasma
    c_median=cm(0.6)  # orange
    c_fill_68=cm(0.2)  # dark purple
    c_fill_95=cm(0.15)  # very dark purple
    c_iwa_narrow=cm(0.85)  # bright yellow-orange
    c_iwa_wide=cm(0.5)  # magenta
    c_orbit_light=cm(0.2)  # dark purple for orbit traces
    c_orbit_best=cm(0.95)  # bright yellow for best orbit
    c_start=cm(0.7)  # orange for start marker
    c_end=cm(0.3)  # purple for end marker
    c_star=cm(0.0)  # dark purple/blue for star

    # IWA/OWA values
    IWA_narrow=155
    OWA_narrow=436
    IWA_wide=450
    OWA_wide=1300

    # Determine if we can plot 2D orbits
    plot_2d=(df_sample is not None and params is not None)

    if plot_2d:
        if figsize is None: figsize=(20,12)
        fig=plt.figure(figsize=figsize)
        gs=fig.add_gridspec(4,2,width_ratios=[1.2,1],hspace=0.3,wspace=0.3)

        # 2d orbit trajectory
        epochs_2d=Time(np.linspace(Time(start_date).mjd,Time(end_date).mjd,100),format="mjd")
        raoff_2d,deoff_2d,best_idx=compute_orbit_for_plotting(
            df_sample,epochs_2d,
            params["basis"],params["m0"],params["m0_err"],
            params["plx"],params["plx_err"],
            params["n_planets"],params["pl_num"],
            override_inc=override_inc,
            override_lan=override_lan,
            inc_mean=params.get("inc_mean"),
            inc_sig=params.get("inc_sig"),
            user_inc_mean=user_inc_mean,
            user_inc_sig=user_inc_sig
        )

        ax_orbit=fig.add_subplot(gs[:,0])

        if user_inc_mean is not None and user_inc_sig is not None:
            inc_str=f'{user_inc_mean:.1f}±{user_inc_sig:.1f}°'
        elif override_inc=="gaussian" and params.get("inc_mean") is not None:
            inc_str=f'Gaussian ({params["inc_mean"]:.1f}±{params["inc_sig"]:.1f}°)'
        elif override_inc is not None:
            inc_str=f'{override_inc}°'
        else:
            inc_str='random'

        lan_str='random' if override_lan is None else f'{override_lan}°'
        ax_orbit.set_title(f'{planet_name}: Orbital Trajectory\n(i={inc_str}, Ω={lan_str})',
                           fontsize=14,fontweight='bold',pad=15)
        ax_orbit.set_xlabel('RA Offset [mas]',fontsize=13,fontweight='bold')
        ax_orbit.set_ylabel('Dec Offset [mas]',fontsize=13,fontweight='bold')

        #IWA/OWA circles
        theta=np.linspace(0,2*np.pi,100)
        ax_orbit.plot(IWA_narrow*np.cos(theta),IWA_narrow*np.sin(theta),
                      color=c_iwa_narrow,lw=3,linestyle='--',label='IWA/OWA (Narrow)',alpha=0.7)
        ax_orbit.plot(OWA_narrow*np.cos(theta),OWA_narrow*np.sin(theta),
                      color=c_iwa_narrow,lw=3,linestyle='--',alpha=0.7)
        ax_orbit.plot(IWA_wide*np.cos(theta),IWA_wide*np.sin(theta),
                      color=c_iwa_wide,lw=3,linestyle='--',label='IWA/OWA (Wide)',alpha=0.5)
        ax_orbit.plot(OWA_wide*np.cos(theta),OWA_wide*np.sin(theta),
                      color=c_iwa_wide,lw=3,linestyle='--',alpha=0.5)

        # sample orbits
        n_samples=min(50,raoff_2d.shape[1])
        sample_indices=np.random.choice(raoff_2d.shape[1],n_samples,replace=False)
        for i in sample_indices:
            ax_orbit.plot(raoff_2d[:,i],deoff_2d[:,i],'-',
                          color=c_orbit_light,alpha=0.2,linewidth=1.5)

        # star <3
        ax_orbit.plot(0,0,'*',color=c_star,markersize=25,label='Star',
                      zorder=15,markeredgecolor='yellow',markeredgewidth=0.5)

        ax_orbit.set_aspect('equal')
        ax_orbit.grid(True,alpha=0.2,linestyle=':')
        ax_orbit.legend(loc='best',fontsize=11,framealpha=0.9)
        ax_orbit.tick_params(axis='both',which='major',labelsize=11)

        # Time series plots on right side
        axes=[fig.add_subplot(gs[i,1]) for i in range(4)]

    else:
        # Create figure with only time series (4 subplots stacked)
        if figsize is None: figsize=(14,12)
        fig,axes=plt.subplots(4,1,figsize=figsize)

    start_year=years[0]
    end_year=years[-1]

    if not plot_2d:
        fig.suptitle(f'{planet_name} - Orbital Parameters ({start_year:.1f} → {end_year:.1f})',
                     fontsize=16,fontweight='bold',y=0.995)
    else:
        fig.suptitle(f'{planet_name} - Orbital Analysis ({start_year:.1f} → {end_year:.1f})',
                     fontsize=16,fontweight='bold',y=0.995)

    # Plot 1: Separation (mas)
    ax1=axes[0]
    min_sep=csv_data['separation_mas_16th'].min()
    max_sep=csv_data['separation_mas_84th'].max()
    ax1.set_title(f'Sky-Projected Angular Separation (1σ: {min_sep:.0f}-{max_sep:.0f} mas)',
                  fontsize=12,pad=10)
    ax1.fill_between(years,
                     csv_data['separation_mas_2.5th'],
                     csv_data['separation_mas_97.5th'],
                     color=c_fill_95,alpha=0.3,label='95% CI')
    ax1.fill_between(years,
                     csv_data['separation_mas_16th'],
                     csv_data['separation_mas_84th'],
                     color=c_fill_68,alpha=0.5,label='68% CI')
    ax1.plot(years,csv_data['separation_mas_median'],'-',
             color=c_median,linewidth=2.5,label='Median',marker='o',markersize=3)

    ax1.axhline(y=IWA_narrow,color=c_iwa_narrow,linestyle='--',linewidth=2.5,
                label='IWA/OWA (Narrow)',alpha=0.7)
    ax1.axhline(y=OWA_narrow,color=c_iwa_narrow,linestyle='--',linewidth=2.5,alpha=0.7)
    ax1.axhline(y=IWA_wide,color=c_iwa_wide,linestyle='--',linewidth=2.5,
                label='IWA/OWA (Wide)',alpha=0.5)
    ax1.axhline(y=OWA_wide,color=c_iwa_wide,linestyle='--',linewidth=2.5,alpha=0.5)

    ax1.set_ylabel('Separation (mas)',fontsize=11,fontweight='bold')
    ax1.grid(True,alpha=0.25,linestyle=':')
    ax1.legend(loc='best',fontsize=9,framealpha=0.9)
    ax1.tick_params(axis='both',which='major',labelsize=10)

    # Plot 2: Orbital Radius (AU)
    ax2=axes[1]
    ax2.set_title('3D Orbital Radius',fontsize=12,pad=10)

    ax2.fill_between(years,
                     csv_data['orbital_radius_au_2.5th'],
                     csv_data['orbital_radius_au_97.5th'],
                     color=c_fill_95,alpha=0.3,label='95% CI')
    ax2.fill_between(years,
                     csv_data['orbital_radius_au_16th'],
                     csv_data['orbital_radius_au_84th'],
                     color=c_fill_68,alpha=0.5,label='68% CI')
    ax2.plot(years,csv_data['orbital_radius_au_median'],'-',
             color=c_median,linewidth=2.5,label='Median',marker='o',markersize=3)

    ax2.set_ylabel('Orbital Radius (AU)',fontsize=11,fontweight='bold')
    ax2.grid(True,alpha=0.25,linestyle=':')
    ax2.legend(loc='best',fontsize=9,framealpha=0.9)
    ax2.tick_params(axis='both',which='major',labelsize=10)

    # Plot 3: Phase Angle (deg)
    ax3=axes[2]
    ax3.set_title('Phase Angle',fontsize=12,pad=10)

    ax3.fill_between(years,
                     csv_data['phase_angle_deg_2.5th'],
                     csv_data['phase_angle_deg_97.5th'],
                     color=c_fill_95,alpha=0.3,label='95% CI')
    ax3.fill_between(years,
                     csv_data['phase_angle_deg_16th'],
                     csv_data['phase_angle_deg_84th'],
                     color=c_fill_68,alpha=0.5,label='68% CI')
    ax3.plot(years,csv_data['phase_angle_deg_median'],'-',
             color=c_median,linewidth=2.5,label='Median',marker='o',markersize=3)

    ax3.set_ylabel('Phase Angle (°)',fontsize=11,fontweight='bold')
    ax3.grid(True,alpha=0.25,linestyle=':')
    ax3.legend(loc='best',fontsize=9,framealpha=0.9)
    ax3.tick_params(axis='both',which='major',labelsize=10)

    # Plot 4: True Anomaly (deg)
    ax4=axes[3]
    ax4.set_title('True Anomaly',fontsize=12,pad=10)

    ax4.fill_between(years,
                     csv_data['true_anomaly_deg_16th'],
                     csv_data['true_anomaly_deg_84th'],
                     color=c_fill_68,alpha=0.5,label='68% CI')
    ax4.plot(years,csv_data['true_anomaly_deg_median'],'-',
             color=c_median,linewidth=2.5,label='Median',marker='o',markersize=3)

    ax4.set_ylabel('True Anomaly (°)',fontsize=11,fontweight='bold')
    ax4.set_xlabel('Year',fontsize=11,fontweight='bold')
    ax4.grid(True,alpha=0.25,linestyle=':')
    ax4.legend(loc='best',fontsize=9,framealpha=0.9)
    ax4.tick_params(axis='both',which='major',labelsize=10)

    # Ensure all x-axes show the same range
    for ax in axes:
        ax.set_xlim(years[0],years[-1])

    plt.tight_layout()

    plot_filename=f"{output_prefix}_orbital_params.{fig_ext}"
    plt.savefig(plot_filename,dpi=300,bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    if show_plots:
        plt.show()
    plt.close('all')


def is_detectable(seps,fc,contrast_curve):
    """
    TODO:
    - Make sure IWA and OWA are handled appropriately
    - Think about linear interpolation for v2
    """
    seps = np.array(seps)
    fc = np.array(fc)

    # get the position of the closest separation in the contrast curve for each given sep
    concurve_seps = contrast_curve[0]
    concurve_fcs = contrast_curve[1]
    args = np.argmin(np.abs(seps[:,np.newaxis]-concurve_seps),axis=1)
    limiting_fcs = concurve_fcs[args]
    return fc >= limiting_fcs




def main():
    warnings.warn("Running this code via the .py script is outdated! Please reference the python notebooks for the latest workflow.")
    
    parser=argparse.ArgumentParser(
        description='Generate CSV files with projected angular separations for RV-detected exoplanets.'
    )

    parser.add_argument('--planet',type=str,required=False,
                        choices=list(orbit_params.keys()),
                        help='Planet to compute separations for')

    parser.add_argument('--start-date',type=str,default=None,
                        help='Start date in YYYY-MM-DD format (default: 2027-01-01)')

    parser.add_argument('--end-date',type=str,default=None,
                        help='End date in YYYY-MM-DD format (default: 2027-06-01)')

    parser.add_argument('--time-interval',type=int,default=None,
                        help='Time interval in days (default: 1)')

    parser.add_argument('--inclination',type=str,default=None,
                        help='Orbital inclination: "random", "gaussian", a fixed value (e.g., "90"), or value with uncertainty (e.g., "90±5") (default: random)')

    parser.add_argument('--nsamp',type=str,default=None,
                        help='Number of posterior samples or "all" (default: all)')

    parser.add_argument('--posterior-dir',type=str,default='orbit_fits',
                        help='Directory containing posterior CSV files (default: orbit_fits)')

    parser.add_argument('--output',type=str,default=None,
                        help='Output CSV filename (default: auto-generated)')

    parser.add_argument('--plot',action='store_true',
                        help='Generate plots of orbital parameters over time')

    args=parser.parse_args()

    # Interactive prompts if arguments not provided
    print("="*60)
    print("Planet Separation CSV Generator")
    print("="*60)

    # Planet selection
    if args.planet is None:
        print("Available planets:")
        for i,(key,name) in enumerate(display_names.items(),1):
            print(f"  {i:2d}. {key:15s} - {name}")
        print()

        while True:
            planet_input=input("Select planet (number or name): ").strip()

            try:
                planet_num=int(planet_input)
                if 1<=planet_num<=len(orbit_params):
                    args.planet=list(orbit_params.keys())[planet_num-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(orbit_params)}")
            except ValueError:
                if planet_input in orbit_params:
                    args.planet=planet_input
                    break
                else:
                    print(f"Invalid planet name. Please try again.")

    if args.start_date is None:
        args.start_date=input("Start date [2027-01-01]: ").strip() or "2027-01-01"
    if args.end_date is None:
        args.end_date=input("End date [2027-06-01]: ").strip() or "2027-06-01"
    if args.time_interval is None:
        time_interval_input=input("Time interval in days [1]: ").strip()
        args.time_interval=int(time_interval_input) if time_interval_input else 1

    params=orbit_params[args.planet]
    has_gaussian_info="inc_mean" in params and "inc_sig" in params

    if args.inclination is None:
        while True:
            if has_gaussian_info:
                print(f"\nInclination options:")
                print(f"  - Press Enter for 'random' (cos i uniform distribution)")
                print(
                    f"  - It appears this system already has an inclination constraint. Type 'gaussian' to sample from Gaussian fit (mean={params['inc_mean']:.1f}°, σ={params['inc_sig']:.1f}°)")
                print(f"  - Type a specific value in degrees (e.g., 90)")
                print(f"  - Type a value with uncertainty (e.g., 90±5 or 90+/-5)")
                args.inclination=input("Inclination [random]: ").strip() or "random"
            else:
                print(f"\nInclination options:")
                print(f"  - Press Enter for 'random' (cos i uniform distribution)")
                print(f"  - Type a specific value in degrees (e.g., 90)")
                print(f"  - Type a value with uncertainty (e.g., 90±5 or 90+/-5)")
                args.inclination=input("Inclination [random]: ").strip() or "random"

            # Validate the input
            try:
                parse_inclination(args.inclination)
                break
            except ValueError as e:
                print(f"\n{e}")
                print("Please try again.")
                continue

    if args.nsamp is None:
        nsamp_input=input("Number of posterior samples (or 'all') [all]: ").strip()
        if nsamp_input.lower()=='all' or nsamp_input=='':
            args.nsamp='all'
        else:
            args.nsamp=int(nsamp_input)
    else:
        if isinstance(args.nsamp,str) and args.nsamp.lower()=='all':
            args.nsamp='all'
        else:
            try:
                args.nsamp=int(args.nsamp)
            except ValueError:
                print(f"Error: nsamp must be a number or 'all'")
                return

    # Ask about plotting if not specified via command line
    if not args.plot:
        plot_input=input("Generate plots? (y/n) [n]: ").strip().lower()
        args.plot=plot_input in ['y','yes']

    output=f'{args.planet}_{args.start_date}_to_{args.end_date}_RVOnly.csv'
    output_dir='.'
    override_lan=0.

    inc_mode,inc_value,inc_uncertainty=parse_inclination(args.inclination)
    inc_params=[inc_value,inc_uncertainty]

    df_samples,csv_data=gen_orbit_csv(args.planet,params,
                                      args.posterior_dir,
                                      output_dir,
                                      args.start_date,
                                      args.end_date,
                                      args.time_interval,
                                      inc_mode,
                                      inc_params,
                                      override_lan,
                                      args.nsamp,
                                      output,
                                      args.plot)


if __name__=="__main__":
    main()