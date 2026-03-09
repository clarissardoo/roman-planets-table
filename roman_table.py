import numpy as np
import pandas as pd
from astropy.time import Time
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit
from astropy import units as u
import matplotlib.pyplot as plt
import os,pickle,warnings,argparse,glob
import os
import json
import EXOSIMS.Prototypes.TargetList
import EXOSIMS.Prototypes.TimeKeeping
import EXOSIMS.Observatory.ObservatoryL2Halo
import pandas as pd
import copy
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from corgietc.corgietc import corgietc
from pathlib import Path
from astroquery.simbad import Simbad
from astropy.time import Time
# Load EXOSIMS configuration for CGI
import os
import json

# Load the CGI specs file
scriptfile=os.path.join(os.environ["CORGIETC_DATA_DIR"],"scripts","CGI_Noise.json")
with open(scriptfile,"r") as f:
    specs=json.loads(f.read())

# Update the modules dictionary to use the HIPfromSimbad star catalog
specs["modules"]["StarCatalog"]="HIPfromSimbad"
#%%
import os
import json
import EXOSIMS.Prototypes.TargetList
import EXOSIMS.Prototypes.TimeKeeping
import EXOSIMS.Observatory.ObservatoryL2Halo
import pandas as pd
import copy
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from corgietc.corgietc import corgietc
from pathlib import Path
from astroquery.simbad import Simbad
from astropy.time import Time

orbit_params={

    "eps_Eri_b":{
        'star':'eps_Eri','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":0.82,"m0_err":0.02,
        "plx":310.5773,"plx_err":0.1355,
        "n_planets":1,"pl_num":1,"g_mag":3.465752,
        "inc_mean":78.810,"inc_sig":29.340,
        "posterior_type":"radvel",
    },

    "ups_And_d":{
        'star':'ups_And','pl_letter':'d',
        "basis":"per tc secosw sesinw k",
        "m0":1.29419667430000,"m0_err":0.04122482369025,
        "plx":74.1940,"plx_err":0.2083,
        "n_planets":3,"pl_num":3,"g_mag":3.966133,
        "inc_mean":23.758,"inc_sig":1.316,
        "posterior_type":"radvel",
    },

    "14_Her_b":{
        'star':'14_Her','pl_letter':'b',
        "m0":0.98,"m0_err":0.04,
        "plx":55.8657,"plx_err":0.0291,
        "n_planets":2,"pl_num":1,"g_mag":6.3830000,
        "posterior_type":"orbitize",

    },
}

def display_name(planet_name):  # Display names for prettier output
    replaced_underscores=" ".join(planet_name.split("_"))
    capitalized=replaced_underscores[0].upper()+replaced_underscores[1:]
    return capitalized


def compute_sep(
        df,epochs,basis=None,m0=None,m0_err=None,plx=None,plx_err=None,n_planets=1,pl_num=1,
        override_inc=None,override_lan=None,inc_mean=None,inc_sig=None,
        user_inc_mean=None,user_inc_sig=None,posterior_type=None
):
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

    chain_len=len(df)
    tau_ref_epoch=58849

    # Set default posterior type
    if posterior_type is None:
        posterior_type='radvel'

    if posterior_type=='orbitize':

        print("Using Orbitize posterior format...")
        # Extract orbital elements directly from posterior
        sma=df[f'sma{pl_num}'].values  # AU
        ecc=df[f'ecc{pl_num}'].values
        inc=np.radians(df[f'inc{pl_num}'].values)*180/np.pi  # Convert to radians
        omega_pl_rad=np.radians(df[f'aop{pl_num}'].values)*180/np.pi  # argument of periastron
        lan=np.radians(df[f'pan{pl_num}'].values)*180/np.pi  # position angle of nodes
        tau=df[f'tau{pl_num}'].values

        # Extract stellar mass (m0)
        if 'm0' in df.columns:
            m_st=df['m0'].values
        elif m0 is not None:
            print(f"Warning: Stellar mass (m0) not in posterior, using m0={m0} from params")
            m_st=np.full(chain_len,m0)
        else:
            raise ValueError("Need stellar mass (m0) in posterior or m0 parameter")

        # Extract planet mass (m1, m2, m3, ...)
        planet_mass_col=f'm{pl_num}'
        if planet_mass_col in df.columns:
            m_pl=df[planet_mass_col].values
        else:
            print(f"Warning: Planet mass ({planet_mass_col}) not found in posterior, using fallback estimate")
            m_pl=None

        # Calculate planet mass if not available
        if m_pl is None:
            # Use Kepler's 3rd law as rough estimate
            period_yr=(sma**3/m_st)**(0.5)
            m_pl=0.001*m_st  # Placeholder - 1 Jupiter mass ~0.001 M_sun
            print(f"Warning: Using placeholder planet mass estimate")

        mtot=m_st+m_pl

        # Get parallax
        if 'plx' in df.columns:
            parallax=df['plx'].values
        elif 'parallax' in df.columns:
            parallax=df['parallax'].values
        elif plx is not None:
            parallax=np.random.normal(plx,plx_err if plx_err is not None else 0.01*plx,size=chain_len)
        else:
            raise ValueError("Need parallax in posterior or plx parameter")

    else:

        print("Using RadVel posterior format...")

        if basis is None:
            raise ValueError("basis parameter required for RadVel posteriors")
        if m0 is None:
            raise ValueError("m0 parameter required for RadVel posteriors")
        if plx is None:
            raise ValueError("plx parameter required for RadVel posteriors")

        myBasis=Basis(basis,n_planets)
        df=myBasis.to_synth(df)

        # convert RadVel posteriors -> orbitize posteriors
        m_st=np.random.normal(m0,m0_err,size=chain_len)
        semiamp=df[f'k{pl_num}'].values
        per_day=df[f'per{pl_num}'].values
        period_yr=per_day/365.25
        ecc=df[f'e{pl_num}'].values
        msini=(
                Msini(semiamp,per_day,m_st,ecc,Msini_units='Earth')*
                (u.M_earth/u.M_sun).to('')
        )

        # median msini for critical inclination
        median_msini=np.median(msini)

        # Handle inclination sampling for RadVel
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
            # Default: uniform in cos(inc) with critical inclination constraint
            crit_incrad=np.arcsin(median_msini/0.08)
            cosi=(2.*np.random.random(size=chain_len)*np.cos(crit_incrad))-1.
            inc=np.arccos(cosi)

        m_pl=msini/np.sin(inc)
        mtot=m_st+m_pl
        sma=(period_yr**2*mtot)**(1/3)
        omega_st_rad=df[f'w{pl_num}'].values
        omega_pl_rad=omega_st_rad+np.pi
        parallax=np.random.normal(plx,plx_err,size=chain_len)

        if override_lan is not None:
            lan=np.full(chain_len,np.radians(override_lan))
        else:
            lan=np.random.random_sample(size=chain_len)*2.*np.pi

        tp_mjd=df[f'tp{pl_num}'].values-2400000.5
        tau=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)

    # ==================================================================
    # COMMON CODE - compute projected separation in mas
    # ==================================================================
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

    # Thiele-Innes constants
    A=sma*(np.cos(omega_pl_rad)*np.cos(lan)-np.sin(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    B=sma*(np.cos(omega_pl_rad)*np.sin(lan)+np.sin(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    F=sma*(-np.sin(omega_pl_rad)*np.cos(lan)-np.cos(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    G=sma*(-np.sin(omega_pl_rad)*np.sin(lan)+np.cos(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    C=sma*np.sin(omega_pl_rad)*np.sin(inc)
    H=sma*np.cos(omega_pl_rad)*np.sin(inc)

    # Compute period for mean motion
    period_yr=(sma**3/mtot)**(0.5)
    per_day=period_yr*365.25

    # Compute tp_mjd from tau if needed
    from orbitize.basis import tau_to_tp
    tp_mjd=tau_to_tp(tau,tau_ref_epoch,period_yr)

    for i in range(n_epochs):
        # Mean anomaly
        n_motion=2*np.pi/per_day  # mean motion (rad/day)
        M=n_motion*(epochs.mjd[i]-tp_mjd)

        # Eccentric anomaly
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
    r_au=np.sqrt(x_mas**2+y_mas**2+z_mas**2)/parallax  # AU
    z_au=z_mas/parallax  # AU

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


def weighted_percentile_nan(data,weights,percentile):
    """Compute weighted percentile, ignoring NaN values per row."""
    result=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        valid=~np.isnan(data[i,:])
        if np.sum(valid)<5:
            result[i]=np.nan
            continue
        d=data[i,valid]
        w=weights[valid]
        w=w/np.sum(w)
        sorted_indices=np.argsort(d)
        sorted_data=d[sorted_indices]
        sorted_weights=w[sorted_indices]
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


def compute_orbit_for_plotting(df,epochs,basis=None,m0=None,m0_err=None,plx=None,plx_err=None,
                               n_planets=1,pl_num=1,override_inc=None,
                               override_lan=None,inc_mean=None,inc_sig=None,
                               user_inc_mean=None,user_inc_sig=None,posterior_type='radvel'):
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
    chain_len=len(df)
    tau_ref_epoch=58849

    if posterior_type=='orbitize':

        print("Plotting Orbitize orbits...")

        # Extract orbital elements directly from posterior
        sma=df[f'sma{pl_num}'].values  # AU
        ecc=df[f'ecc{pl_num}'].values
        inc=np.radians(df[f'inc{pl_num}'].values)*180/np.pi  # Convert to radians
        omega_pl_rad=np.radians(df[f'aop{pl_num}'].values)*180/np.pi  # argument of periastron
        lan=np.radians(df[f'pan{pl_num}'].values)*180/np.pi  # position angle of nodes
        tau=df[f'tau{pl_num}'].values

        # Extract stellar mass
        if 'm0' in df.columns:
            m_st=df['m0'].values
        elif m0 is not None:
            m_st=np.full(chain_len,m0)
        else:
            raise ValueError("Need stellar mass (m0) in posterior or m0 parameter")

        # Extract planet mass
        planet_mass_col=f'm{pl_num}'
        if planet_mass_col in df.columns:
            m_pl=df[planet_mass_col].values
        else:
            # Fallback estimate
            print(f"Warning: Planet mass ({planet_mass_col}) not in posterior, using estimate")
            m_pl=0.001*m_st  # ~1 Jupiter mass

        mtot=m_st+m_pl

        # Get parallax
        if 'plx' in df.columns:
            parallax=df['plx'].values
        elif 'parallax' in df.columns:
            parallax=df['parallax'].values
        elif plx is not None:
            parallax=np.random.normal(plx,plx_err if plx_err is not None else 0.01*plx,size=chain_len)
        else:
            raise ValueError("Need parallax in posterior or plx parameter")

        # Get best-fit index based on chi2 (lower is better)
        if 'chi2' in df.columns:
            best_idx=np.argmin(df['chi2'].values)
        else:
            best_idx=0  # Default to first sample if no chi2

    else:
        # RADVEL POSTERIORS - convert to orbitize format
        print("Plotting RadVel orbits...")

        if basis is None:
            raise ValueError("basis parameter required for RadVel posteriors")
        if m0 is None:
            raise ValueError("m0 parameter required for RadVel posteriors")
        if plx is None:
            raise ValueError("plx parameter required for RadVel posteriors")

        myBasis=Basis(basis,n_planets)
        df=myBasis.to_synth(df)

        # convert RadVel posteriors -> orbitize posteriors
        m_st=np.random.normal(m0,m0_err,size=chain_len)
        semiamp=df[f'k{pl_num}'].values
        per_day=df[f'per{pl_num}'].values
        period_yr=per_day/365.25
        ecc=df[f'e{pl_num}'].values
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
        omega_st_rad=df[f'w{pl_num}'].values
        omega_pl_rad=omega_st_rad+np.pi
        parallax=np.random.normal(plx,plx_err,size=chain_len)

        if override_lan is not None:
            lan=np.full(chain_len,np.radians(override_lan))
        else:
            lan=np.random.random_sample(size=chain_len)*2.*np.pi

        tp_mjd=df[f'tp{pl_num}'].values-2400000.5
        tau=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)

        # Get best-fit index based on lnprobability (higher is better)
        lnlike=df["lnprobability"].values
        best_idx=np.argmax(lnlike)

    # COMMON CODE - compute projected separation
    raoff,deoff,vz=calc_orbit(
        epochs.mjd,sma,ecc,inc,
        omega_pl_rad,lan,tau,
        parallax,mtot,tau_ref_epoch=tau_ref_epoch
    )

    return raoff,deoff,best_idx


def load_point_cloud(planet,
                     i_dir='.',
                     start_date='2027-01-01',
                     end_date='2027-06-30',
                     fname=None,
                     broadcast_arrays=True,
                     ):
    # Load pickle
    if fname is None:
        planet_name=planet.replace("_","")
        fname=f"{planet_name}_{start_date}_to_{end_date}_PointCloud.pkl"
    fpath=os.path.join(i_dir,fname)
    print(f'Loading point cloud from {fpath}')
    if os.path.exists(fpath):
        with open(fpath,'rb') as f:
            point_cloud=pickle.load(f)
    else:
        raise FileNotFoundError

    # recast data to consistent array size
    if broadcast_arrays:
        arr_shape=point_cloud['sep_mas'].shape
        for param,arr in point_cloud.items():
            if arr.shape==(arr_shape[0],):
                point_cloud[param]=np.full(arr_shape,arr[:,np.newaxis])
            elif arr.shape==(arr_shape[1],):
                point_cloud[param]=np.full(arr_shape,arr)
            elif arr.shape!=arr_shape:
                raise ValueError(
                    f'param {param} has unexpected array shape {arr.shape}, when sep_mas has shape {arr_shape}.')

    return point_cloud


def load_posteriors(planet,params=None,
                    posterior_dir='orbit_fits',
                    format="radvel"
                    ):
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
    import glob

    if params is None:
        params=orbit_params[planet]
    star=params['star']

    if format=='radvel':
        planet_dir=os.path.join(posterior_dir,star)
        files=list(glob.glob(os.path.join(planet_dir,"*.csv.bz2")))

        if len(files)==0:
            raise UserWarning(f"Error: No posterior data found for {planet} in {planet_dir}")
        if len(files)>1:
            raise UserWarning(f"Multiple posterior data files found for {planet} in {planet_dir}")

        print(f"Loading RadVel posterior from {files[0]}...")

        df=pd.read_csv(files[0])

    elif format=='orbitize':
        # For orbitize, files are in Roman_RV_HGCA_Orbits subdirectory
        planet_dir=os.path.join(posterior_dir,'Roman_RV_HGCA_Orbits',star)

        # Look for CSV files (both .csv and .csv.bz2)
        files=list(glob.glob(os.path.join(planet_dir,"*.csv")))
        files+=list(glob.glob(os.path.join(planet_dir,"*.csv.bz2")))

        if len(files)==0:
            raise UserWarning(f"Error: No posterior data found for {planet} in {planet_dir}")
        if len(files)>1:
            # Try to find the file that matches the planet letter
            pl_letter=params.get('pl_letter','b')
            matching_files=[f for f in files if pl_letter in os.path.basename(f).lower()]
            if len(matching_files)==1:
                files=matching_files
            else:
                raise UserWarning(f"Multiple posterior data files found for {planet} in {planet_dir}: {files}")

        print(f"Loading Orbitize posterior from {files[0]}...")
        df=pd.read_csv(files[0])

    else:
        raise UserWarning(f"Unknown posterior format: {format}")

    return df


def get_likelihood_weights(df,posterior_type='radvel'):
    """
    Extract likelihood weights from posterior DataFrame.

    Args:
        df (pd.DataFrame): Posterior samples
        posterior_type (str): 'radvel' or 'orbitize'

    Returns:
        np.array: Normalized weights for each sample
    """

    if posterior_type=='orbitize':
        # Orbitize uses chi2
        if 'chi2' in df.columns:
            chi2=df['chi2'].values
            # Convert chi2 to likelihood: L = exp(-chi2/2)
            # Then normalize
            log_like=-chi2/2
            weights=np.exp(log_like-np.max(log_like))
            weights=weights/np.sum(weights)
        else:
            print("Warning: No chi2 column found, using uniform weights")
            weights=np.ones(len(df))/len(df)

    else:  # radvel
        if 'lnprobability' in df.columns:
            lnlike=df['lnprobability'].values
            weights=np.exp(lnlike-np.max(lnlike))
            weights=weights/np.sum(weights)
        else:
            print("Warning: No lnprobability column found, using uniform weights")
            weights=np.ones(len(df))/len(df)

    return weights


def gen_point_cloud(planet,post_df,
                    params=None,  #override default planet params
                    output_dir='.',
                    start_date='2027-01-01',
                    end_date='2027-06-30',
                    time_interval=1,
                    inc_mode='random',
                    inc_params=None,
                    override_lan=0.,
                    nsamp='all',
                    out_fname=None,
                    standard_arr_size=False,
                    posterior_type='radvel'):
    if nsamp=='all':
        nsamp=len(post_df)
        print(f"Using all {nsamp} posterior samples")

    print()
    print("-"*60)
    print(f"Configuration:")
    print(f"  Planet: {display_name(planet)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Time interval: {time_interval} days")

    if params is None:
        params=orbit_params[planet]

    if posterior_type=='orbitize':
        print(f"  Inclination: from Orbitize posterior (sampled)")
        print(f"  Ω (Omega): from Orbitize posterior (sampled)")
        override_inc=None
        user_inc_mean=None
        user_inc_sig=None
    elif inc_mode=='user_gaussian':
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
    df_sample=post_df.sample(nsamp,
                             replace=True)  # TODO: Consider if replace=True is the right choice statistically? -Ell

    n_epochs=int((t_end.mjd-t_start.mjd)/time_interval)+1
    epochs=Time(np.linspace(t_start.mjd,t_end.mjd,n_epochs),format="mjd")

    print(f"Generating point cloud for {n_epochs} epochs...")

    # output shape of each array is (n_epochs,nsamp)
    seps_mas,raoff_mas,deoff_mas,m_pl,inc,true_anomaly,z_au,r_au,parallax=compute_sep(
        df_sample,epochs,
        params.get("basis"),params["m0"],params["m0_err"],
        params["plx"],params["plx_err"],
        params["n_planets"],params["pl_num"],
        override_lan=override_lan,
        override_inc=override_inc,
        inc_mean=params.get("inc_mean"),
        inc_sig=params.get("inc_sig"),
        user_inc_mean=user_inc_mean,
        user_inc_sig=user_inc_sig,
        posterior_type=posterior_type
    )

    # Phase angle calculation using 3D radius
    phase_angle_rad=np.arccos(z_au/r_au)
    phase_angle_deg=np.degrees(phase_angle_rad)

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

    if posterior_type=='orbitize':
        if 'chi2' in df_sample.columns:
            lnlike=-df_sample['chi2'].values/2
        else:
            lnlike=np.zeros(len(df_sample))
    else:
        myBasis=Basis(params["basis"],params["n_planets"])
        df_synth=myBasis.to_synth(df_sample)
        lnlike=df_synth["lnprobability"].values

    # Organize output cloud
    epochs=epochs.value

    if standard_arr_size:
        distance_pc=np.full_like(seps_mas,1000.0/parallax)
        lnlike=np.full_like(seps_mas,lnlike)  # ln likelihood
        epochs=np.full_like(seps_mas,epochs[:,np.newaxis])
        m_pl_mjup=np.full_like(seps_mas,m_pl_mjup)
        r_pl_rjup=np.full_like(seps_mas,r_pl_rjup)
        inc_deg=np.full_like(seps_mas,inc_deg)

    point_cloud={
        'epoch_mjd':epochs,
        'sep_mas':seps_mas,
        'raoff_mas':raoff_mas,
        'deoff_mas':deoff_mas,
        'true_anom_deg':true_anomaly,
        'z_au':z_au,
        'orbital_radius_au':r_au,
        'phase_angle_deg':phase_angle_deg,
        'm_pl_mjup':m_pl_mjup,
        'r_pl_rjup':r_pl_rjup,
        'inc_deg':inc_deg,
        'ln_likelihood':lnlike,
        'parallax_mas':parallax
    }

    # Save pickle
    if out_fname is None:
        planet_name=planet.replace("_","")
        output_file=f"{planet_name}_{start_date}_to_{end_date}_PointCloud.pkl"
    else:
        output_file=out_fname.split('.')[0]+'_PointCloud.pkl'
    output_fpath=os.path.join(output_dir,output_file)
    print(f'Saving point cloud to {output_fpath}')
    with open(output_fpath,'wb') as f:
        pickle.dump(point_cloud,f)

    return point_cloud


def calc_integration_time_for_cloud(
    planet,
    point_cloud,
    target_snr=5,
    obs_mode='IMG_NFB1_HLC',
    band=1,
    params=None,
    n_inttime_samples=None,
    launch_date=None,
    IWA_mas=None,
    OWA_mas=None,
    max_inttime_hours=1000,
    min_det_prob=0.05,
):
    import time

    if params is None:
        params = orbit_params[planet]

    if launch_date is None:
        launch_date = Time('2026-09-28T00:00:00.000', format='isot')

    # Build target list from star HIP number
    star = params['star']
    HIP_name = Simbad.query_objectids(star, criteria="ident.id LIKE 'HIP%'")['id'][0]
    HIP_num = int(HIP_name.split('HIP')[1])
    TL = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs), catalogpath=[HIP_num])
    OS = TL.OpticalSystem
    sInd = 0
    mode_opt = list(filter(lambda m: m['Scenario'] == f'OPT_{obs_mode}', OS.observingModes))[0]
    mode_con = list(filter(lambda m: m['Scenario'] == f'CON_{obs_mode}', OS.observingModes))[0]
    mode_opt['SNR'] = target_snr
    mode_con['SNR'] = target_snr

    # IWA/OWA from mode if not provided
    if IWA_mas is None:
        IWA_mas = mode_opt['syst']['IWA'].to(u.mas).value
    if OWA_mas is None:
        OWA_mas = mode_opt['syst']['OWA'].to(u.mas).value

    # Build TK once outside epoch loop (not passed to calc_intTime anyway)
    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)


    # Epochs
    if point_cloud['epoch_mjd'].ndim == 2:
        epoch_mjd = point_cloud['epoch_mjd'][:, 0]
    else:
        epoch_mjd = point_cloud['epoch_mjd']
    n_epochs = len(epoch_mjd)

    # Sample indices (this is for faster computation - for robustness use all but note it'll take a long time D: )
    # we favor high lnlike from point cloud here
    total_samples=point_cloud['sep_mas'].shape[1]

    lnlike=point_cloud['ln_likelihood']
    if lnlike.ndim==2:
        lnlike=lnlike[0]

    w=np.exp(lnlike-np.max(lnlike))
    w=w/np.sum(w)

    if n_inttime_samples is None or n_inttime_samples=='all':
        sample_indices=np.arange(total_samples)
    else:
        n_inttime_samples=min(int(n_inttime_samples),total_samples)
        sample_indices=np.random.choice(
            total_samples,
            size=n_inttime_samples,
            replace=False,
            p=w
        )
    n_samples=len(sample_indices)

    # Output arrays
    Tint_opt_hours = np.full((n_epochs, n_samples), np.nan)
    Tint_con_hours = np.full((n_epochs, n_samples), np.nan)

    # det prob mask to save computation time
    has_det_prob = 'detection_probability' in point_cloud

    print(f'Calculating integration times for {planet}, band {band}')
    print(f'  {n_epochs} epochs x {n_samples} samples')
    print(f'  IWA={IWA_mas:.1f} mas, OWA={OWA_mas:.1f} mas')
    print(f'  Max integration time: {max_inttime_hours} hours')
    print()

    scenario_start = time.time()
    epochs_computed = 0  # track epochs where we actually called calc_intTime
    for i_epoch in range(n_epochs):
        # Progress every epoch
        elapsed = time.time() - scenario_start
        rate = epochs_computed / elapsed if elapsed > 0 else 0
        remaining = (n_epochs - i_epoch) / rate if rate > 0 else 0
        epoch_mjd_val = epoch_mjd[i_epoch]
        epoch_year = Time(epoch_mjd_val, format='mjd').decimalyear
        print(f'  Epoch {i_epoch+1:3d}/{n_epochs} '
              f'(year={epoch_year:.3f}) | '
              f'{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining | '
              f'{epochs_computed} computed, {i_epoch - epochs_computed} skipped',
              end='\r')
        # Skip epoch if det prob too low
        if has_det_prob:
            det_prob = point_cloud['detection_probability'][i_epoch]
            if det_prob < min_det_prob:
                continue  # leave as NaN

        # Per-sample arrays for this epoch
        seps          = point_cloud['sep_mas'][i_epoch, sample_indices]
        contrasts_raw = point_cloud['flux_contrast'][i_epoch, sample_indices]
        orb_rad       = point_cloud['orbital_radius_au'][i_epoch, sample_indices]

        # Safety: mask zero/negative contrasts before log
        valid_contrast = contrasts_raw > 0
        contrasts_dmag = np.where(valid_contrast, -2.5 * np.log10(contrasts_raw), np.nan)

        # mask invalid contrasts and separations
        valid=np.isfinite(contrasts_dmag)
        valid&=(seps>=IWA_mas)&(seps<=OWA_mas)
        if not np.any(valid):
            continue  # leave as NaN

        seps_v     = seps[valid]
        contrast_v = contrasts_dmag[valid]
        orb_rad_v  = orb_rad[valid]
        n_valid    = int(np.sum(valid))

        # Per-sample exozodi scaled by orbital radius
        JEZ_v   = TL.JEZ0[mode_opt["hex"]][sInd] / orb_rad_v**2
        fZ_v    = np.repeat(TL.ZodiacalLight.fZ0, n_valid)
        sInds_v = np.zeros(n_valid, dtype=int)

        for mode, arr in [(mode_opt, Tint_opt_hours), (mode_con, Tint_con_hours)]:
            try:
                Tint = OS.calc_intTime(
                    TL,
                    sInds_v,
                    fZ_v,
                    JEZ_v,
                    contrast_v,
                    seps_v * u.mas,
                    mode
                )
                t_hours = Tint.to_value(u.h)
                t_hours[t_hours > max_inttime_hours] = np.nan
                arr[i_epoch, valid] = t_hours
            except Exception as e:
                print(f'\n  Warning: calc_intTime failed at epoch {i_epoch}: {e}')
                continue

        epochs_computed += 1

    print(f'\nDone, Total time: {(time.time() - scenario_start)/60:.1f} minutes')
    print(f'  {epochs_computed}/{n_epochs} epochs computed, '
          f'{n_epochs - epochs_computed} skipped (det_prob or outside FOV)')

    # Report
    for label, arr in [('opt', Tint_opt_hours), ('con', Tint_con_hours)]:
        valid_vals = arr[np.isfinite(arr)]
        n_nan = np.sum(np.isnan(arr))
        print(f'  {label}: {len(valid_vals)} valid, {n_nan} NaN'
              + (f', range {valid_vals.min():.2f}-{valid_vals.max():.2f} h '
                 f'(median {np.median(valid_vals):.2f} h)'
                 if len(valid_vals) > 0 else ''))

    point_cloud['integration_time_hours_opt'] = Tint_opt_hours
    point_cloud['integration_time_hours_con'] = Tint_con_hours
    point_cloud['integration_time_days_opt']  = Tint_opt_hours / 24.0
    point_cloud['integration_time_days_con']  = Tint_con_hours / 24.0
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


def load_summary_csv(planet,
                     i_dir='.',
                     start_date='2027-01-01',
                     end_date='2027-06-30',
                     output=None):
    if output is None:
        planet_name=planet.replace("_","")
        csv_file=f"{planet_name}_separations_{start_date}_to_{end_date}.csv"
    else:
        csv_file=output.split('.')[0]+'.csv'

    csv_fpath=os.path.join(i_dir,csv_file)
    print(f"Reading orbit summary csv from {csv_fpath}...")
    csv_data=pd.read_csv(csv_fpath)

    return csv_data


def gen_summary_csv(planet,
                    point_cloud,
                    output_dir='.',
                    output=None):
    """
    Generate summary CSV with weighted statistics including integration times.
    """

    m_pl_mjup=point_cloud['m_pl_mjup']
    mass_median=np.median(m_pl_mjup)
    mass_16th=np.percentile(m_pl_mjup,16)
    mass_84th=np.percentile(m_pl_mjup,84)
    mass_err_lower=mass_median-mass_16th
    mass_err_upper=mass_84th-mass_median

    r_pl_rjup=point_cloud['r_pl_rjup']
    rad_median=np.median(r_pl_rjup)
    rad_16th=np.percentile(r_pl_rjup,16)
    rad_84th=np.percentile(r_pl_rjup,84)
    rad_err_lower=rad_median-rad_16th
    rad_err_upper=rad_84th-rad_median

    inc_deg=point_cloud['inc_deg']
    inc_median=np.median(inc_deg)
    inc_16th=np.percentile(inc_deg,16)
    inc_84th=np.percentile(inc_deg,84)

    print(f"Planet mass: {mass_median:.2f} +{mass_err_upper:.2f}/-{mass_err_lower:.2f} M_Jup")
    print(f"Planet radius: {rad_median:.2f} +{rad_err_upper:.2f}/-{rad_err_lower:.2f} R_Jup")
    print(f"Inclination: {inc_median:.2f} [{inc_16th:.2f}, {inc_84th:.2f}] degrees")
    print()

    # Get epochs
    if point_cloud['epoch_mjd'].ndim==2:
        epochs=Time(point_cloud['epoch_mjd'][:,0],format='mjd')
    else:
        epochs=Time(point_cloud['epoch_mjd'],format='mjd')

    start_date=epochs.iso[0][:10]
    end_date=epochs.iso[-1][:10]

    # Initialize CSV data dictionary
    csv_data_dict={
        'date_iso':epochs.iso,
        'mjd':epochs.mjd,
        'decimal_year':epochs.decimalyear,
    }

    # Add optional scalar fields
    if 'detection_probability' in point_cloud.keys():
        csv_data_dict['det_probability']=point_cloud['detection_probability']
    if 'GB_not_observable' in point_cloud.keys():
        csv_data_dict['GB_not_observable']=point_cloud['GB_not_observable']
    if 'targ_observable' in point_cloud.keys():
        csv_data_dict['targ_observable']=point_cloud['targ_observable']

    # Prepare phase angle calculations
    phase_angle_rad=point_cloud['phase_angle_deg']*np.pi/180.
    lambert_phase=(np.sin(phase_angle_rad)+(np.pi-phase_angle_rad)*
                   np.cos(phase_angle_rad))/np.pi

    # Dictionary of labeled data to compute statistics for
    labeled_data={
        'separation_mas':point_cloud['sep_mas'],
        'orbital_radius_au':point_cloud['orbital_radius_au'],
        'phase_angle_deg':point_cloud['phase_angle_deg'],
        'lambert_phase':lambert_phase,
        'true_anomaly':np.degrees(point_cloud['true_anom_deg'])%360,
    }

    # Add optional keys
    optional_keys=['phi_x_a','flux_contrast']
    for key in optional_keys:
        if key in point_cloud.keys():
            labeled_data[key]=point_cloud[key]

    # Weight the posteriors by lnlike for FULL sample arrays
    lnlike=point_cloud['ln_likelihood']
    if lnlike.ndim==2:
        lnlike=lnlike[0]
    weights=np.exp(lnlike-np.max(lnlike))
    weights=weights/np.sum(weights)

    # Compute percentiles, mean, and std for all quantities
    for label,arr in labeled_data.items():
        csv_data_dict[f'{label}_median']=weighted_percentile(arr,weights,50)
        csv_data_dict[f'{label}_16th']=weighted_percentile(arr,weights,16)
        csv_data_dict[f'{label}_84th']=weighted_percentile(arr,weights,84)
        csv_data_dict[f'{label}_2.5th']=weighted_percentile(arr,weights,2.5)
        csv_data_dict[f'{label}_97.5th']=weighted_percentile(arr,weights,97.5)
        csv_data_dict[f'{label}_mean']=weighted_mean(arr,weights)
        csv_data_dict[f'{label}_std']=weighted_std(arr,weights)

    # Handle integration times separately (they have different sample sizes)
    # Use weighted_percentile_nan so that out-of-FOV NaNs propagate correctly
    # into the CSV — epochs where the planet is inside the IWA will have NaN
    # percentiles, which in turn produces gaps in the integration time plot.
    if 'integration_time_hours_opt' in point_cloud.keys():
        if 'integration_time_sample_indices' in point_cloud.keys():
            inttime_indices=point_cloud['integration_time_sample_indices']
            inttime_weights=weights[inttime_indices]
            inttime_weights=inttime_weights/np.sum(inttime_weights)
        else:
            inttime_weights=weights

        for key in ['integration_time_hours_opt','integration_time_hours_con',
                    'integration_time_days_opt','integration_time_days_con']:
            if key in point_cloud.keys():
                arr=point_cloud[key]
                csv_data_dict[f'{key}_median']=weighted_percentile_nan(arr,inttime_weights,50)
                csv_data_dict[f'{key}_16th']=weighted_percentile_nan(arr,inttime_weights,16)
                csv_data_dict[f'{key}_84th']=weighted_percentile_nan(arr,inttime_weights,84)
                csv_data_dict[f'{key}_2.5th']=weighted_percentile_nan(arr,inttime_weights,2.5)
                csv_data_dict[f'{key}_97.5th']=weighted_percentile_nan(arr,inttime_weights,97.5)
                csv_data_dict[f'{key}_mean']=weighted_mean(arr,inttime_weights)
                csv_data_dict[f'{key}_std']=weighted_std(arr,inttime_weights)

    # Create DataFrame
    csv_data=pd.DataFrame(csv_data_dict)

    # Determine output file name
    if output is None:
        planet_name=planet.replace("_","")
        output_file=f"{planet_name}_separations_{start_date}_to_{end_date}.csv"
    else:
        output_file=output.split('.')[0]+'.csv'

    output_fpath=os.path.join(output_dir,output_file)
    print(f"Writing output to {output_fpath}...")
    with open(output_fpath,'w') as f:
        csv_data.to_csv(f,index=False)

    return csv_data


def plot_orbital_parameters(planet,csv_data,output_prefix,
                            df_sample=None,params=None,override_inc=None,
                            override_lan=None,user_inc_mean=None,user_inc_sig=None,
                            start_date=None,end_date=None,figsize=None,fig_ext='png',
                            band=None,posterior_type='radvel',show_WFOV=False,
                            show_plots=False):
    """
    Create plots including 2D orbits and time-series parameters.
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
    c_opt=cm(0.3)  # color for optimistic integration time
    c_con=cm(0.7)  # color for conservative integration time

    # IWA/OWA values
    IWA_narrow=155
    OWA_narrow=436
    IWA_wide=300
    OWA_wide=994
    OWA=OWA_wide if show_WFOV else OWA_narrow

    # Determine if we can plot 2D orbits
    plot_2d=(df_sample is not None and params is not None)

    # Set plot location indices
    n_param_plots=2
    sep=0
    phase=1

    if 'det_probability' in csv_data.columns:
        n_param_plots+=1
        sep+=1
        phase+=1
        det=0
        plot_det=True
    else:
        plot_det=False

    if 'flux_contrast_median' in csv_data.columns:
        n_param_plots+=1
        fc=phase+1
        plot_fc=True
    else:
        plot_fc=False

    # Check if integration time data exists
    plot_inttime=('integration_time_hours_opt_median' in csv_data.columns and
                  'integration_time_hours_con_median' in csv_data.columns)

    if plot_inttime:
        n_param_plots+=1
        inttime_idx=n_param_plots-1

    if plot_2d:
        if figsize is None:
            figsize=(20,12)
        fig=plt.figure(figsize=figsize)
        gs=fig.add_gridspec(n_param_plots,2,width_ratios=[1.2,1],
                            hspace=0.3,wspace=0.3)

        # 2D orbit trajectory
        epochs_2d=Time(np.linspace(Time(start_date).mjd,Time(end_date).mjd,100),
                       format="mjd")
        raoff_2d,deoff_2d,best_idx=compute_orbit_for_plotting(
            df_sample,epochs_2d,
            params.get("basis"),params["m0"],params["m0_err"],
            params["plx"],params["plx_err"],
            params["n_planets"],params["pl_num"],
            override_inc=override_inc,
            override_lan=override_lan,
            inc_mean=params.get("inc_mean"),
            inc_sig=params.get("inc_sig"),
            user_inc_mean=user_inc_mean,
            user_inc_sig=user_inc_sig,
            posterior_type=posterior_type
        )

        ax_orbit=fig.add_subplot(gs[:,0])

        # Generate title strings based on posterior type
        if posterior_type=='orbitize':
            inc_col=f'inc{params["pl_num"]}'
            pan_col=f'pan{params["pl_num"]}'

            if inc_col in df_sample.columns:
                inc_median=np.median(df_sample[inc_col])*180/np.pi
                inc_16=np.percentile(df_sample[inc_col],16)*180/np.pi
                inc_84=np.percentile(df_sample[inc_col],84)*180/np.pi
                inc_str=f'{inc_median:.1f}° [{inc_16:.1f}°-{inc_84:.1f}°]'
            else:
                inc_str='from posterior'

            if pan_col in df_sample.columns:
                pan_median=np.median(df_sample[pan_col])*180/np.pi
                pan_16=np.percentile(df_sample[pan_col],16)*180/np.pi
                pan_84=np.percentile(df_sample[pan_col],84)*180/np.pi
                lan_str=f'{pan_median:.1f}° [{pan_16:.1f}°-{pan_84:.1f}°]'
            else:
                lan_str='from posterior'
        else:
            if user_inc_mean is not None and user_inc_sig is not None:
                inc_str=f'{user_inc_mean:.1f}±{user_inc_sig:.1f}°'
            elif override_inc=="gaussian" and params.get("inc_mean") is not None:
                inc_str=f'Gaussian ({params["inc_mean"]:.1f}±{params["inc_sig"]:.1f}°)'
            elif override_inc is not None:
                inc_str=f'{override_inc}°'
            else:
                inc_str='random'

            lan_str='random' if override_lan is None else f'{override_lan}°'

        ax_orbit.set_title(f'{display_name(planet)}: Orbital Trajectory\n(i={inc_str}, Ω={lan_str})',
                           fontsize=14,fontweight='bold',pad=15)
        ax_orbit.set_xlabel('RA Offset [mas]',fontsize=13,fontweight='bold')
        ax_orbit.set_ylabel('Dec Offset [mas]',fontsize=13,fontweight='bold')

        # IWA/OWA circles
        theta=np.linspace(0,2*np.pi,100)
        ax_orbit.plot(IWA_narrow*np.cos(theta),IWA_narrow*np.sin(theta),
                      color=c_iwa_narrow,lw=3,linestyle='--',
                      label='IWA/OWA (Narrow)',alpha=0.7)
        ax_orbit.plot(OWA_narrow*np.cos(theta),OWA_narrow*np.sin(theta),
                      color=c_iwa_narrow,lw=3,linestyle='--',alpha=0.7)
        ax_orbit.plot(IWA_wide*np.cos(theta),IWA_wide*np.sin(theta),
                      color=c_iwa_wide,lw=3,linestyle='--',
                      label='IWA/OWA (Wide)',alpha=0.5)
        ax_orbit.plot(OWA_wide*np.cos(theta),OWA_wide*np.sin(theta),
                      color=c_iwa_wide,lw=3,linestyle='--',alpha=0.5)

        # Sample orbits
        n_samples=min(50,raoff_2d.shape[1])
        sample_indices=np.random.choice(raoff_2d.shape[1],n_samples,replace=False)
        for i in sample_indices:
            ax_orbit.plot(raoff_2d[:,i],deoff_2d[:,i],'-',
                          color=c_orbit_light,alpha=0.2,linewidth=1.5)

        # Star
        ax_orbit.plot(0,0,'*',color=c_star,markersize=25,label='Star',
                      zorder=15,markeredgecolor='yellow',markeredgewidth=0.5)

        xlims=(np.array([1.1,1.1])*OWA)
        ax_orbit.set_xlim(*xlims)
        ax_orbit.set_ylim(*xlims)

        ax_orbit.set_aspect('equal')
        ax_orbit.grid(True,alpha=0.2,linestyle=':')
        ax_orbit.legend(loc='best',fontsize=11,framealpha=0.9)
        ax_orbit.tick_params(axis='both',which='major',labelsize=11)

        # Time series plots on right side
        axes=[fig.add_subplot(gs[i,1]) for i in range(n_param_plots)]

    else:
        # Create figure with only time series (stacked subplots)
        if figsize is None:
            figsize=(14,12)
        fig,axes=plt.subplots(n_param_plots,1,figsize=figsize)
        if n_param_plots==1:
            axes=[axes]

    start_year=years[0]
    end_year=years[-1]

    band_str=f' Band {band}' if not (band is None) else ''
    fig.suptitle(f'{display_name(planet)}{band_str} ({start_year:.1f} → {end_year:.1f})',
                 fontsize=16,fontweight='bold',y=0.995)

    # Plot 1: Separation (mas)
    ax1=axes[sep]
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

    if show_WFOV:
        ax1.axhline(y=IWA_wide,color=c_iwa_wide,linestyle='--',linewidth=2.5,
                    label='IWA/OWA (Wide)',alpha=0.5)
        ax1.axhline(y=OWA_wide,color=c_iwa_wide,linestyle='--',linewidth=2.5,alpha=0.5)
    else:
        ax1.axhline(y=IWA_narrow,color=c_iwa_narrow,linestyle='--',linewidth=2.5,
                    label='IWA/OWA (Narrow)',alpha=0.7)
        ax1.axhline(y=OWA_narrow,color=c_iwa_narrow,linestyle='--',linewidth=2.5,alpha=0.7)

    ax1.set_ylabel('Separation (mas)',fontsize=11,fontweight='bold')
    ax1.grid(True,alpha=0.25,linestyle=':')
    ax1.legend(loc='best',fontsize=9,framealpha=0.9)
    ax1.tick_params(axis='both',which='major',labelsize=10)
    ax1.set_ylim(0,OWA*1.1)

    # Plot 2: Phase Angle (deg)
    ax3=axes[phase]
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
    ax3.set_ylim(0,180)
    ax3.grid(True,alpha=0.25,linestyle=':')
    ax3.legend(loc='best',fontsize=9,framealpha=0.9)
    ax3.tick_params(axis='both',which='major',labelsize=10)

    # Optional Plot: Detection Probability
    if plot_det:
        ax_detprob=axes[det]
        ax_detprob.plot(years,csv_data['det_probability'],'-',
                        color=c_median,linewidth=2.5,marker='o',markersize=3)

        ax_detprob.set_ylabel('Detection Probability',fontsize=11,fontweight='bold')
        ax_detprob.set_ylim(0,1)
        ax_detprob.grid(True,alpha=0.25,linestyle=':')
        ax_detprob.tick_params(axis='both',which='major',labelsize=10)

    # Optional Plot: Flux Contrast
    if plot_fc:
        ax_fc=axes[fc]
        ax_fc.fill_between(years,
                           csv_data['flux_contrast_2.5th'],
                           csv_data['flux_contrast_97.5th'],
                           color=c_fill_95,alpha=0.3,label='95% CI')
        ax_fc.fill_between(years,
                           csv_data['flux_contrast_16th'],
                           csv_data['flux_contrast_84th'],
                           color=c_fill_68,alpha=0.5,label='68% CI')
        ax_fc.plot(years,csv_data['flux_contrast_median'],'-',
                   color=c_median,linewidth=2.5,label='Median',marker='o',markersize=3)
        ax_fc.set_yscale('log')
        ax_fc.set_ylabel('Flux Contrast',fontsize=11,fontweight='bold')
        ax_fc.set_ylim(1e-9,1e-7)
        ax_fc.grid(True,alpha=0.25,linestyle=':')
        ax_fc.legend(loc='best',fontsize=9,framealpha=0.9)
        ax_fc.tick_params(axis='both',which='major',labelsize=10)

    # Optional Plot: Integration Time
    if plot_inttime:
        ax_int=axes[inttime_idx]

        # mask NaNs so fill_between leaves gaps
        valid_mask_opt=np.isfinite(csv_data['integration_time_hours_opt_median'].values)
        valid_mask_con=np.isfinite(csv_data['integration_time_hours_con_median'].values)

        if np.any(valid_mask_opt) or np.any(valid_mask_con):

            #OPTIMISTIC
            if np.any(valid_mask_opt):
                y_med=csv_data['integration_time_hours_opt_median'].values.copy()

                # 68%
                y16=csv_data['integration_time_hours_opt_16th'].values.copy()
                y84=csv_data['integration_time_hours_opt_84th'].values.copy()

                # 95%
                y025=csv_data['integration_time_hours_opt_2.5th'].values.copy()
                y975=csv_data['integration_time_hours_opt_97.5th'].values.copy()

                # apply gaps
                for arr in (y_med,y16,y84,y025,y975):
                    arr[~valid_mask_opt]=np.nan

                # plot 95% behind 68%
                ax_int.fill_between(years,y025,y975,
                                    color=c_opt,alpha=0.15,label='Optimistic 95% CI')
                ax_int.fill_between(years,y16,y84,
                                    color=c_opt,alpha=0.30,label='Optimistic 68% CI')
                ax_int.plot(years,y_med,'-',color=c_opt,linewidth=2.5,
                            label='Optimistic median',marker='o',markersize=3)

            #CONSERVATIVE
            if np.any(valid_mask_con):
                y_med=csv_data['integration_time_hours_con_median'].values.copy()

                # 68%
                y16=csv_data['integration_time_hours_con_16th'].values.copy()
                y84=csv_data['integration_time_hours_con_84th'].values.copy()

                # 95%
                y025=csv_data['integration_time_hours_con_2.5th'].values.copy()
                y975=csv_data['integration_time_hours_con_97.5th'].values.copy()

                for arr in (y_med,y16,y84,y025,y975):
                    arr[~valid_mask_con]=np.nan

                ax_int.fill_between(years,y025,y975,
                                    color=c_con,alpha=0.15,label='Conservative 95% CI')
                ax_int.fill_between(years,y16,y84,
                                    color=c_con,alpha=0.30,label='Conservative 68% CI')
                ax_int.plot(years,y_med,'-',color=c_con,linewidth=2.5,
                            label='Conservative median',marker='o',markersize=3)

            ax_int.set_ylabel('Integration Time (hours)',fontsize=11,fontweight='bold')
            ax_int.set_yscale('log')

            # sensible y-lims using medians
            all_valid=[]
            if np.any(valid_mask_opt):
                all_valid.extend(csv_data['integration_time_hours_opt_median'].values[valid_mask_opt])
            if np.any(valid_mask_con):
                all_valid.extend(csv_data['integration_time_hours_con_median'].values[valid_mask_con])
            if len(all_valid)>0:
                min_val=np.nanmin(all_valid)
                max_val=np.nanmax(all_valid)
                ax_int.set_ylim(min_val*0.5,max_val*2.0)

            ax_int.grid(True,alpha=0.25,linestyle=':')
            ax_int.legend(loc='best',fontsize=9,framealpha=0.9)
            ax_int.tick_params(axis='both',which='major',labelsize=10)

        else:
            ax_int.text(0.5,0.5,'No integration time data available',
                        ha='center',va='center',transform=ax_int.transAxes,
                        fontsize=12,color='gray')
            ax_int.set_ylabel('Integration Time (hours)',fontsize=11,fontweight='bold')
    # Add observation windows LAST (so they don't affect ylims)
    for a,ax in enumerate(axes):
        ylims=ax.get_ylim()

        if 'GB_not_observable' in csv_data.columns:
            label='GB Observations' if a==0 else ''
            ax.fill_between(years,ylims[0],ylims[1],
                            where=~csv_data.GB_not_observable,
                            alpha=0.15,edgecolor='None',color='orange',
                            label=label,zorder=0)

        if 'targ_observable' in csv_data.columns:
            label='Solar Keepout' if a==0 else ''
            ax.fill_between(years,ylims[0],ylims[1],
                            where=~csv_data.targ_observable,
                            alpha=0.15,edgecolor='None',color='gray',
                            label=label,zorder=0)

        ax.set_ylim(ylims)

        if a==0 and ('GB_not_observable' in csv_data.columns or
                     'targ_observable' in csv_data.columns):
            ax.legend(loc='best',fontsize=9,framealpha=0.9)

        ax.set_xlim(years[0],years[-1])

    axes[-1].set_xlabel('Year',fontsize=11,fontweight='bold')

    plt.tight_layout()

    plot_filename=f"{output_prefix}_orbital_params.{fig_ext}"
    plt.savefig(plot_filename,dpi=150,bbox_inches='tight')
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
    seps=np.array(seps)
    fc=np.array(fc)

    # get the position of the closest separation in the contrast curve for each given sep
    concurve_seps=contrast_curve[0]
    concurve_fcs=contrast_curve[1]
    if seps.ndim==1:
        args=np.argmin(np.abs(seps[np.newaxis]-concurve_seps),axis=1)
    elif seps.ndim==2:
        args=np.argmin(np.abs(seps[:,:,np.newaxis]-concurve_seps),axis=2)
    else:
        raise UserWarning('separation and flux contrast should be 1D or 2D array.')
    limiting_fcs=concurve_fcs[args]
    return fc>=limiting_fcs


def main():
    warnings.warn(
        "Running this code via the .py script is outdated! Please reference the python notebooks for the latest workflow.")

    parser=argparse.ArgumentParser(
        description='Generate CSV files with projected angular separations for RV-detected exoplanets.'
    )

    parser.add_argument('--planet',type=str,required=False,
                        choices=list(orbit_params.keys()),
                        help='Planet to compute separations for')

    parser.add_argument('--start-date',type=str,default=None,
                        help='Start date in YYYY-MM-DD format (default: 2027-01-01)')

    parser.add_argument('--end-date',type=str,default=None,
                        help='End date in YYYY-MM-DD format (default: 2027-06-30)')

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

    print("="*60)
    print("Planet Separation CSV Generator")
    print("="*60)

    if args.planet is None:
        print("Available planets:")
        for i,(key,name) in enumerate(orbit_params.items(),1):
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

    if not args.plot:
        plot_input=input("Generate plots? (y/n) [n]: ").strip().lower()
        args.plot=plot_input in ['y','yes']

    output=f'{args.planet}_{args.start_date}_to_{args.end_date}_RVOnly.csv'
    output_dir='.'
    override_lan=0.

    inc_mode,inc_value,inc_uncertainty=parse_inclination(args.inclination)
    inc_params=[inc_value,inc_uncertainty]


if __name__=="__main__":
    main()