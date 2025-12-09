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


def compute_sep(
        df,epochs,basis,m0,m0_err,plx,plx_err,n_planets=1,pl_num=1,
        override_inc=None,override_lan=None
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

    Returns:
        tuple of:
            np.array of size (len(epochs) x len(df)): sky-projected angular
                separations [mas] at each input epoch
            np.array: RA offsets [mas]
            np.array: Dec offsets [mas]
            np.array: planet masses in solar masses
            np.array: inclinations in radians
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

    if override_inc is not None:
        inc=np.full(chain_len,np.radians(override_inc))
    else:
        cosi=(2.*np.random.random(size=chain_len))-1.
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
    raoff,deoff,_=calc_orbit(
        epochs.mjd,sma,ecc,inc,
        omega_pl_rad,lan,tau,
        parallax,mtot,tau_ref_epoch=tau_ref_epoch
    )
    seps=np.sqrt(raoff**2+deoff**2)

    return seps,raoff,deoff,m_pl,inc


def weighted_percentile(data,weights,percentile):
    """
    Compute weighted percentile along axis=1
    data: shape (n_epochs, n_samples)
    weights: shape (n_samples,)
    percentile: float (0-100)
    """
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
        "n_planets":1,"pl_num":1,"g_mag":3.465752,
    },
    "HD_87883":{
        "basis":"per tc secosw sesinw k",
        "m0":0.810,"m0_err":0.091,
        "plx":54.6421000,"plx_err":0.0369056,
        "n_planets":1,"pl_num":1,"g_mag":7.286231,
    },
    "HD_114783":{
        "basis":"per tc secosw sesinw k",
        "m0":0.90,"m0_err":0.04,
        "plx":47.4482000,"plx_err":0.0637202,
        "n_planets":2,"pl_num":2,"g_mag":7.330857,
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
        "n_planets":1,"pl_num":1,"g_mag":6.583667,
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
        "n_planets":2,"pl_num":1,"g_mag":5.552787
    },
    "HD_217107":{
        "basis":"per tc secosw sesinw k",
        "m0":1.05963082882500,"m0_err":0.04470613802572,
        "plx":49.8170000,"plx_err":0.0573616,
        "n_planets":2,"pl_num":2,"g_mag":5.996743
    },
    "pi_Men":{
        "basis":"per tc secosw sesinw k",
        "m0":1.10,"m0_err":0.14,
        "plx":54.705200,"plx_err":0.067131,
        "n_planets":1,"pl_num":1,"g_mag":5.511580
    },
    "ups_And":{
        "basis":"per tc secosw sesinw k",
        "m0":1.29419667430000,"m0_err":0.04122482369025,
        "plx":74.571100,"plx_err":0.349118,
        "n_planets":3,"pl_num":3,"g_mag":3.966133
    },
    "HD_192310":{
        "basis":"per tc secosw sesinw k",
        "m0":0.84432448757250,"m0_err":0.02820926681885,
        "plx":113.648000,"plx_err":0.118606,
        "n_planets":2,"pl_num":2,"g_mag":5.481350
    },
}


def main():
    parser=argparse.ArgumentParser(
        description='Generate CSV files with projected angular separations for RV-detected exoplanets.'
    )

    parser.add_argument('--planet',type=str,required=False,
                        choices=list(orbit_params.keys()),
                        help='Planet to compute separations for')

    parser.add_argument('--start-date',type=str,default=None,
                        help='Start date in YYYY-MM-DD format (default: 2027-01-01)')

    parser.add_argument('--end-date',type=str,default=None,
                        help='End date in YYYY-MM-DD format (default: 2031-06-01)')

    parser.add_argument('--time-interval',type=int,default=None,
                        help='Time interval in days (default: 30)')

    parser.add_argument('--inclination',type=str,default=None,
                        help='Orbital inclination in degrees or "random" (default: random)')

    parser.add_argument('--nsamp',type=str,default=None,
                        help='Number of posterior samples or "all" (default: all)')

    parser.add_argument('--posterior-dir',type=str,default='orbit_fits',
                        help='Directory containing posterior CSV files (default: orbit_fits)')

    parser.add_argument('--output',type=str,default=None,
                        help='Output CSV filename (default: auto-generated)')

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

            # Try by number
            try:
                planet_num=int(planet_input)
                if 1<=planet_num<=len(orbit_params):
                    args.planet=list(orbit_params.keys())[planet_num-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(orbit_params)}")
            except ValueError:
                # Try by name
                if planet_input in orbit_params:
                    args.planet=planet_input
                    break
                else:
                    print(f"Invalid planet name. Please try again.")


    # DATES
    # Start date
    if args.start_date is None:
        args.start_date=input("Start date [2027-01-01]: ").strip() or "2027-01-01"
    # End date
    if args.end_date is None:
        args.end_date=input("End date [2031-06-01]: ").strip() or "2031-06-01"
    # Time interval
    if args.time_interval is None:
        time_interval_input=input("Time interval in days [30]: ").strip()
        args.time_interval=int(time_interval_input) if time_interval_input else 30

    #Orbit Posteriors
    # Inclination
    if args.inclination is None:
        args.inclination=input("Inclination in degrees or 'random' [random]: ").strip() or "random"
    # Number of samples
    if args.nsamp is None:
        nsamp_input=input("Number of posterior samples (or 'all') [all]: ").strip()
        if nsamp_input.lower()=='all' or nsamp_input=='':
            args.nsamp='all'
        else:
            args.nsamp=int(nsamp_input)
    else:
        # Handle command-line argument (could be string "all" or number)
        if isinstance(args.nsamp,str) and args.nsamp.lower()=='all':
            args.nsamp='all'
        else:
            try:
                args.nsamp=int(args.nsamp)
            except ValueError:
                print(f"Error: nsamp must be a number or 'all'")
                return

    # check posterior size if "all" is requested
    base_path=Path(args.posterior_dir)
    planet_dir=base_path/args.planet
    files=list(planet_dir.glob("*.csv.bz2"))
    if not files:
        print(f"Error: No posterior data found for {args.planet} in {planet_dir}")
        return

    print(f"Loading posterior data from {files[0]}...")
    df=pd.read_csv(files[0])
    params=orbit_params[args.planet]
    if args.nsamp=='all':
        args.nsamp=len(df)
        print(f"Using all {args.nsamp} posterior samples")

    print()
    print("-"*60)
    print(f"Configuration:")
    print(f"  Planet: {display_names[args.planet]}")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Time interval: {args.time_interval} days")
    print(f"  Inclination: {args.inclination}")
    print(f"  Posterior samples: {args.nsamp}")
    print("-"*60)
    print()

    # Parse inc
    try:
        override_inc=None if args.inclination.lower()=='random' else float(args.inclination)
        override_lan=None  # Always random
    except ValueError:
        print(f"Error: Invalid inclination value")
        return

    # Parse dates
    try:
        t_start=Time(args.start_date)
        t_end=Time(args.end_date)
    except:
        print("Error: Invalid date format. Use YYYY-MM-DD (e.g., 2026-06-01)")
        return
    if t_end<=t_start:
        print("Error: End date must be after start date")
        return

    # Sample from posteriors
    print(f"Sampling {args.nsamp} orbits from posterior...")
    df_sample=df.sample(args.nsamp,replace=True)

    # Create epochs based on time interval
    n_epochs=int((t_end.mjd-t_start.mjd)/args.time_interval)+1
    epochs=Time(np.linspace(t_start.mjd,t_end.mjd,n_epochs),format="mjd")

    print(f"Computing separations for {n_epochs} epochs...")

    # Compute separations
    try:
        seps,raoff,deoff,m_pl,inc=compute_sep(
            df_sample,epochs,
            params["basis"],params["m0"],params["m0_err"],
            params["plx"],params["plx_err"],
            params["n_planets"],params["pl_num"],
            override_inc=override_inc,
            override_lan=override_lan
        )
    except Exception as e:
        print(f"Error computing separations: {e}")
        return

    # Convert planet mass from solar masses to Jupiter masses
    m_pl_mjup=m_pl*(u.M_sun/u.M_jup).to('')

    # Convert inclination from radians to degrees
    inc_deg=np.degrees(inc)

    # mass and inclination statistics
    mass_median=np.median(m_pl_mjup)
    mass_16th=np.percentile(m_pl_mjup,16)
    mass_84th=np.percentile(m_pl_mjup,84)
    mass_err_lower=mass_median-mass_16th
    mass_err_upper=mass_84th-mass_median

    inc_median=np.median(inc_deg)
    inc_16th=np.percentile(inc_deg,16)
    inc_84th=np.percentile(inc_deg,84)

    print(f"Planet mass: {mass_median:.2f} +{mass_err_upper:.2f}/-{mass_err_lower:.2f} M_Jup")
    print(f"Inclination: {inc_median:.2f} [{inc_16th:.2f}, {inc_84th:.2f}] degrees")
    print()

    # Get lnlike for weighting the posteriors
    myBasis=Basis(params["basis"],params["n_planets"])
    df_synth=myBasis.to_synth(df_sample)
    lnlike=df_synth["lnprobability"].values

    # Convert lnlike to weights
    weights=np.exp(lnlike-np.max(lnlike))
    weights=weights/np.sum(weights)

    # Compute weighted statistics
    med_sep=weighted_percentile(seps,weights,50)
    low_sep=weighted_percentile(seps,weights,16)
    high_sep=weighted_percentile(seps,weights,84)
    low_sep_95=weighted_percentile(seps,weights,2.5)
    high_sep_95=weighted_percentile(seps,weights,97.5)

    # Convert angular separation (mas) to physical separation (AU)

    distance_pc=1000.0/params["plx"]
    med_rad_au=med_sep*distance_pc/1000.0  # convert mas*pc to AU
    low_rad_au=low_sep*distance_pc/1000.0
    high_rad_au=high_sep*distance_pc/1000.0
    low_rad_au_95=low_sep_95*distance_pc/1000.0
    high_rad_au_95=high_sep_95*distance_pc/1000.0

    # Create DataFrame for CSV
    csv_data=pd.DataFrame({
        'date_iso':epochs.iso,
        'mjd':epochs.mjd,
        'decimal_year':epochs.decimalyear,
        'separation_mas_median':med_sep,
        'separation_mas_16th':low_sep,
        'separation_mas_84th':high_sep,
        'separation_mas_2.5th':low_sep_95,
        'separation_mas_97.5th':high_sep_95,
        'separation_au_median':med_rad_au,
        'separation_au_16th':low_rad_au,
        'separation_au_84th':high_rad_au,
        'separation_au_2.5th':low_rad_au_95,
        'separation_au_97.5th':high_rad_au_95,
    })

    # output file name
    if args.output is None:
        planet_name=args.planet.replace("_","")
        output_file=f"{planet_name}_separations_{args.start_date}_to_{args.end_date}.csv"
    else:
        output_file=args.output

    #write csv
    print(f"Writing output to {output_file}...")
    with open(output_file,'w') as f:
        f.write(f"# Planet: {display_names[args.planet]}\n")
        f.write(f"# Date range: {args.start_date} to {args.end_date}\n")
        f.write(f"# Time interval: {args.time_interval} days\n")
        f.write(f"# Inclination: {args.inclination}\n")
        f.write(f"# Number of posterior samples: {args.nsamp}\n")
        f.write(f"# Number of epochs: {n_epochs}\n")
        f.write(f"#\n")
        f.write(f"# System parameters:\n")
        f.write(f"# Distance: {distance_pc:.2f} pc (parallax: {params['plx']:.2f} +/- {params['plx_err']:.2f} mas)\n")
        f.write(f"#\n")
        f.write(f"# Derived parameters:\n")
        f.write(f"# Planet mass: {mass_median:.3f} +{mass_err_upper:.3f}/-{mass_err_lower:.3f} M_Jup\n")
        f.write(
            f"# Inclination distribution: {inc_median:.2f} deg (median), [{inc_16th:.2f}, {inc_84th:.2f}] deg (16th-84th percentile)\n")
        f.write("#\n")
        csv_data.to_csv(f,index=False)

    print(f"Output saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Planet: {display_names[args.planet]}")
    print(f"  Distance: {distance_pc:.2f} pc")
    print(f"  Epochs: {n_epochs}")
    print(
        f"  Separation range: {med_sep.min():.2f} - {med_sep.max():.2f} mas ({med_rad_au.min():.2f} - {med_rad_au.max():.2f} AU)")


if __name__=="__main__":
    main()