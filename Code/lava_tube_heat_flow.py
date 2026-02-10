"""
1D heat flow model. Can calculate heat flow for basalt on the Moon or Earth. On the Moon, a two layer model (regolith and basalt)
is used. The model will automatically find the largest possible timestep that satisfies numerical stability conditions. Note that
each celestial body will take time to come to equilibrium depending on thickness of rock and regolith, so play with the day value
some and inspect the gif output.

Author: Tyler Paladino

Run from command line. 
Options:
  --Earth               Set Earth to True
  --Moon                Set Moon to True
  --roof_thickness      Set tube roof thickness in meters | required
  --regolith_thickness  Set regolith thickness in meters | required
  --nx                  Set number of grid points in model domain | required
  --lower_boundary_type Set lower boundary type. Input 'fixed' to fix lower boundary at specific temperature (Dirichelet BC). 
                        Must also input temperature 'T_lower'. Input 'insulated' to assume no flux out of bottom of model (Neumann BC) | required
  --T_lower             Set lower boundary temperature in K if using 'fixed' boundary condition 
  --T_basalt            Set temperature in K of starting column | required
  --days                Number of days to run model for | required
  --starting_date       Give starting date in format YYYY-MM-DD_HH:MM. Timezone is assumed to be UTC | required
  --repeat              Whether to repeat chunk of time throughout simulation. Options are: 'day' for repeating 1st 24 hours, 'year' for repeating 1st 365 days, or 'none' for no repetition. Default is 'none' | required 
  --plot_interval       Interval in timesteps (give as integer) to plot data. Lower values will give smoother gifs but will take more time to run. Default is 250 | required
  --ERA5_input_path     Path to ERA5 reanalysis timeseries data containing downwelling solar flux and 2 meter temperature data. Download at CDS archive: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview" | required

Ex for Earth:
python lava_tube_heat_flow.py --Earth --roof_thickness 5 --regolith_thickness 0 --nx 50 --lower_boundary_type insulated --T_basalt 300 --days 365 --starting_date 2020-06-02_06:00 --repeat day --plot_interval 250 --ERA5_input_path data/Tabernacle_ERA5_ts.nc

This will run an earth model for 365 days with a tube roof thickness of 5 meters, no reoglith, a lower boundary type of insulated, 
a starting basalt temperature of 300 K, a starting date of June 2nd, 2020 at 6:00 UTC, using ERA5 data from the Tabernacle area. It will also repeat the first day over and over to create a constant timeseries and will plot every 250 timesteps.

Output will be in a folder that is named by the date and time the script was run in the output directory
Script will generate a .gif and two .csv files:
    -surf_temp.csv: Contains time, surface temperature, incoming solar flux, and outgoing radiative flux
    -run_parameters.csv: Contains all input parameters for the run

Required packages:
imageio
matplotlib
numpy
pandas
tqdm
xarray
"""

import argparse
import io
import os
import warnings

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import xarray as xr

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Earth",
                        help="Set Earth to True", 
                        action="store_true")

    parser.add_argument("--Moon", 
                        help="Set Moon to True", 
                        action="store_true")

    parser.add_argument("--roof_thickness", 
                        help="Set tube roof thickness in meters", 
                        required=True)

    parser.add_argument("--regolith_thickness", 
                        help="Set regolith thickness in meters", 
                        required=True)

    parser.add_argument("--nx", 
                        help="Override number of grid points in model domain. Otherwise, model automatically chooses based on domain length and assuming grid spacing of 0.1 m. Note that setting this value manually may lead to numerical instability issues.", 
                        required=False)

    parser.add_argument("--lower_boundary_type",
                        help="Set lower boundary type. Input 'fixed' to fix lower boundary at specific temperature (Dirichelet BC). Must also input temperature 'T_lower'. Input 'insulated' to assume no flux out of bottom of model (Neumann BC)",
                        required=True)

    parser.add_argument("--T_lower",
                        required=False,
                        help="Set lower boundary temperature in K if using 'fixed' boundary condition")

    parser.add_argument("--T_basalt", 
                        required=True, 
                        help="Set temperature in Kof starting column.")

    parser.add_argument("--days", 
                        help="Number of days to run model for", 
                        required=True)

    parser.add_argument("--starting_date",
                        help="Give starting date in format YYYY-MM-DD_HH:MM. Timezone is assumed to be UTC",
                        required=True)
    
    parser.add_argument("--repeat",
                        help="Whether to repeat chunk of time throughout simulation. Options are: 'day' for repeating 1st 24 hours, 'year' for repeating 1st 365 days, or 'none' for no repetition. Default is 'none'.",
                        required=True,
                        default="none")
    
    parser.add_argument("--plot_interval",
                        help = "Interval in timesteps (give as integer) to plot data. Lower values will give smoother gifs but will take more time to run. Default is 250.",
                        required=True,
                        default=250)
    
    parser.add_argument("--ERA5_input_path",
                        help="Path to ERA5 reanalysis timeseries data containing downwelling solar flux and 2 meter temperature data. Download at CDS archive: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview",
                        required=True)
    


    inputs = parser.parse_args()

    return inputs


def calc_K(temp, vacuum=False):
    """Calculates thermal diffusivity using the emperical formula
    k = A+ B/T + CT^3 from Fujii et al. 1973. A, B, and C parameters are
    for basalt (also from Fujii et al. 1973)

    Parameters
    ----------
    temp : float
        Temperature value at some depth
    vacuum : bool, optional
        Whether the calculation is in vacuum or not, by default False

    Returns
    -------
    therm_diffusivity : float
        Thermal Diffusivity in units of m^2/s
    """
    if vacuum == False:
        A = 3.59e-3  # (cm^2/s)
        B = 1.65  # (cm^2 K/s)
        C = -0.68e-12  # (cm^2/s K^3)
    elif vacuum == True:
        A = 3.03e-3  # (cm^2/s)
        B = 1.30  # (cm^2 K/s)
        C = 1.93e-12  # (cm^2/s K^3)
    therm_diffusivity = A + (B / temp) + (C * temp**3)  # cm^2/s
    # print(therm_diffusivity/10000)
    return therm_diffusivity / 10000


def calc_K_regolith(conductivity, cp, density=1100):
    """Calculates thermal diffusivity from conductivity, specific heat,
    and density

    Parameters
    ----------
    conductivity : float
        Thermal conductivity
    cp : float
        Specific heat
    density : float
        Bulk rock density, by default 1100 (kg/m^3) (Hayne et al. 2013)

    Returns
    -------
    float
        _description_
    """
    return conductivity / (density * cp)


def calc_conductivity_lunar(T, Kc=3.4e-3, chi=2.7):
    """Calculate thermal conductivity based on eq. A4 from Hayne et al. 2017

    Parameters
    ----------
    T : array
        Temperature array (K)
    Kc : float
        Phonon conductivity, by default 3.4e-3 (W m^-1 K^-1)
    chi : float
        Radiative conductivity parameter, by default 2.7 (unitless)

    Returns
    -------
    float
        Thermal conductivity (W m^-1 K^-1)
    """
    return Kc * (1 + chi * (T / 350) ** 3)


def calc_conductivity_earth(T):

    # Calculate temperature dependent thermal conductivity using relation from Haenel and Zoth (1973)
    # and found to be accurate for basalt by Halbert and Parnell (2022)
    return 3.6 - (0.49e-2 * T) + (0.61e-5 * T**2) + (2.58e-9 * T**3)


def calc_cp(T, c0=-3.6125, c1=2.7431, c2=2.3616e-3, c3=-1.234e-5, c4=8.9093e-9):
    """Calculate specific heat using heat capacity coefficients from
    Hayne et al. (2017) eq. A6 and table A1 using equations from Hemingway et al. (1981)
    and Ledlow et al. (1992)

    Parameters
    ----------
    T : array
        Temperature array (K)
    c0 : float, optional
        c0 specific heat equation coefficient, by default -3.6125 (J kg^-1 K^-1)
    c1 : float, optional
        c1 specific heat equation coefficient, by default 2.7431 (J kg^-1 K^-2)
    c2 : _type_, optional
        c2 specific heat equation coefficient, by default 2.3616e-3 J (kg^-1 K^-3)
    c3 : _type_, optional
        c3 specific heat equation coefficient, by default -1.234e-5 J (kg^-1 K^-4)
    c4 : _type_, optional
        c4 specific heat equation coefficient, by default 8.9093e-9 J (kg^-1 K^-5)

    Returns
    -------
    float
        Specific heat capacity (J kg^-1 K^-1)
    """
    return c0 + (c1 * T) + (c2 * T**2) + (c3 * T**3) + (c4 * T**4)

def find_reg_depth_idx(roof_thick, dom_len, dom_arr):
    """Creates a boolean that contains 1 for basalt and
    0 for regolith. Calculates what proportion of the domain is
    basalt or regolith

    Parameters
    ----------
    roof_thick : int
        Roof thickness
    dom_len : int
        Length of domain
    dom_arr : array
        grid array

    Returns
    -------
    bool
        Regolith boolean list
    """
    # Set list up to track what material we're in
    # 1 = basalt, 0 = regolith
    reg_bool = np.ones_like(dom_arr)

    # If roof thickness is the same as domain length, there is no regolith
    # Return bool of only 1s.
    if roof_thick == dom_len:
        return reg_bool
    else:
        # Set from 0th element to correct portion of x that has regolith equal to 0
        reg_bool[len(dom_arr) - round(len(dom_arr) / (dom_len / roof_thick)) :: -1] = 0

        return reg_bool


def calc_dtmax(dx_step, therm_diff):
    """Calculates the maximum timestep value based off the spatial resolution
    and the thermal diffusivity

    Parameters
    ----------
    dx_step : float
        Spatial resolution in meters
    therm_diff : float
        Thermal diffusivity in m^2/s

    Returns
    -------
    float
        Maximum timestep to not be exceeded
    """
    return (dx_step**2) / (2 * therm_diff)


def find_best_dt(del_x, temp_min, temp_max, args):
    """Precompute possible timestep values and find the largest timestep that satisfies numerical conditions.

    Parameters
    ----------
    lower_t_bound : float
        lower boundary temperature [K]
    del_x : float
        spatial resolution [m]
    temp_min : float
        minimum temperature in entire model domain [K]
    temp_max : float
        Maximum temperature in entire model domain [K]
    args : Namespace
        Parsed in arguments

    Returns
    -------

    best_dt : float
        Best dt value (largest without violating any numerical conditions)
    """
    # Generate range of timesteps to test in log space.
    dt_range_test = np.logspace(1, 9, 1000)
    print("Finding optimal dt value")

    # Create list of temperatures based on lowest to highest temps in domain
    T_list = np.linspace(temp_min, temp_max, 100000)

    if args.Earth:
        # Calculate K for all temperatutes in T_list
        K_rock = calc_K(T_list, vacuum=False)
        # Calculate maximum allowable timestep of every K value
        dtmax_rock = calc_dtmax(del_x, K_rock)

        # create boolean array that shows which values on test timestep array are less than the min dt max value
        rock_bool = dt_range_test <= np.min(dtmax_rock)

        # Get indices in rock_bool that are True, return last element
        best_dt_idx = np.where(rock_bool)[0][-1]

    elif args.Moon:
        # Calculate K for all rock temperatutes in T_list
        K_rock = calc_K(T_list, vacuum=True)
        # Calculate K for all regolith temperatutes in T_list
        K_reg = calc_K_regolith(calc_conductivity_lunar(T_list), calc_cp(T_list))

        # Calculate maximum allowable timestep of every K value for rock
        dtmax_rock = calc_dtmax(del_x, K_rock)
        # create boolean array that shows which values on test timestep array are less than the min dt max value
        rock_bool = dt_range_test <= np.min(dtmax_rock)

        # Calculate maximum allowable timestep of every K value for reg
        dtmax_reg = calc_dtmax(del_x, K_reg)
        # create boolean array that shows which values on test timestep array are less than the min dt max value for reg
        reg_bool = dt_range_test <= np.min(dtmax_reg)

        # Get indices in rock_bool and reg_bool that are True, return last element
        best_dt_idx = np.where(rock_bool & reg_bool)[0][-1]

    best_dt = 5 * np.floor(dt_range_test[best_dt_idx] / 5)
    return best_dt - 1000


def calculate_radiative_heat_flux(T_ambient, T_surf, epsilon=0.95):
    """Calculates radiative heat flux

    Args:
        T_surf (float): Surface temperature (K)
        epsilon (float, optional): Emissivity. Defaults to 0.72. Emissivity from Mineo and Pappalardo


    Returns:
        Q_rad (float): Radiative heat flux (W m^-2)
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)

    # Calculate radiative heat flux out of the surface using Stephan-Boltzmann law
    Q_rad = sigma * epsilon * (T_surf**4 - T_ambient**4)  # Radiative heat flux at the surface

    return Q_rad

def calculate_convective_heat_flux(T_ambient,T_surf,h):
    """Calculates convective heat flux above surface

    Args:
        T_ambient (float): Ambient temperature (K)
        T_surf (float): Surface temperature (K)
        h (float): Convective heat transfer coefficient (W m^-2 K^-1) 

    Returns:
        (float): Convective heat flux (W m^-2)
    """    
    return h * (T_surf - T_ambient) 

def get_ERA5_at_ts(ds, timestep, start_time,repeat):
    """Gets net flux at current timestep and location

    Args:
        ds (xarray): ERA5 reanalysis data containing previously calculated net flux data
        timestep (int): current timestep in seconds
        start_time (datetime): starting time of simulation (UTC)
        lat (float): latitude of location
        lon (float): longitude of location

    Returns:
        float: Net heat flux at the surface (W/m^2) at the given location and timestep
    """ 
    if repeat == "day":
        # If repeating by day, find the time within the first day by taking the modulus of the timestep with the number of seconds in a day
        timestep = timestep % (24*60*60)
    elif repeat == "year":
        # If repeating by year, find the time within the first year by taking the modulus of the timestep with the number of seconds in a year
        timestep = timestep % (365*24*60*60)
    else:
        # If not repeating, use the full timestep value to find the current time
        pass

    # Find current time by taking start time input by user and adding the current timestep value
    time_now = start_time + datetime.timedelta(seconds=timestep)
    
    # Index into dataset to get net flux value at current time (or closest time if exact time can't be found)
    return ds.sel(valid_time=time_now,method='nearest')

def calculate_solar_heat_flux(
    lat, lon, alt, start_time, day_repeat_num, dt_freq, total_time=1
):
    """Calculates solar heating flux for one day starting on start_time. Repeats this day over day_repeat_num
    to create a constant timeseries. Uses pvlib to calculate clearsky irradiance.

    Args:
        lat (float): latitude of location
        lon (float): longitude of location
        alt (float): altitude of location
        start_time (datetime): datetime object for starting time of the one day period
        day_repeat_num (int): number of times to repeat the one day period
        dt_freq (int): time frequency in seconds for solar flux data
        total_time (int,optional): total time in days for the repeated data. Defaults to 1.


    Returns:
        clearsky_repeatday (pd.DataFrame): DataFrame with repeated clearsky irradiance data
    """
    # Create date range for total time at given dt frequency
    times = pd.date_range(
        start_time,
        start_time + datetime.timedelta(days=total_time),
        freq=str(dt_freq) + "s",
        tz="UTC",
    )

    # Create location object, and get solar position and clearsky irradiance data.
    location = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    clearsky = location.get_clearsky(times)

    # Repeat solar flux data for specified number of days
    clearsky_repeatday = pd.concat([clearsky] * day_repeat_num, ignore_index=True)
    full_year_times = pd.date_range(
        start_time, periods=len(clearsky_repeatday), freq=str(dt_freq) + "s", tz="UTC"
    )
    clearsky_repeatday.index = full_year_times

    return clearsky_repeatday

def main():
    ############################# Define inputs ########################################

    cmd_inputs = parse()
    # Regolith thickness
    regolith_thickness = float(cmd_inputs.regolith_thickness)  # Should be 0 for Earth

    # Tube roof thickness
    tube_roof = float(cmd_inputs.roof_thickness)

    # Days to run model for
    days = int(cmd_inputs.days)
    # Convert to seconds
    total_time = 24 * 60 * 60 * days

    # Starting datetime
    start_dt = datetime.datetime.strptime(cmd_inputs.starting_date, "%Y-%m-%d_%H:%M")

    # Total length of domain in meters
    L = regolith_thickness + tube_roof

    # Number of grid points
    

    #  Space array + grid spacing
    if cmd_inputs.nx is None:
        dx= 0.1
        x = np.arange(0,L+dx,dx)
        nx = len(x)
    else:
        nx = int(cmd_inputs.nx)
        x, dx = np.linspace(0, L, nx, retstep=True)

    # Find which parts of domain are regolith
    regolith_bool = find_reg_depth_idx(tube_roof, L, x)

    # Bulk rock temperature.
    T_basalt = float(cmd_inputs.T_basalt)

    ######################## Compute time & temperature arrays ############################

    # Precompute smallest timestep value that satisfies numerical requirements
    dt = find_best_dt(dx, 225, 350, cmd_inputs)
    # dt=3600.0

    # Create time array based on dt
    nt = np.arange(1, total_time, dt)
    print("dt=" + str(dt))
    print("dx=" + str(dx))

    # Create temperature array to model on. Make same length as x domain. Set all values equal to basalt temp to start
    T = np.ones_like(x) * T_basalt
    
    # Read in ERA5 reanalysis timeseries data
    ERA5_ds = xr.open_dataset(cmd_inputs.ERA5_input_path)

    # Check if simulation time is too long for ERA5 dataset
    if start_dt + datetime.timedelta(days=days) > pd.to_datetime(ERA5_ds.valid_time.max().data).to_pydatetime():
        raise ValueError(f"The end date of the simulation ({start_dt + datetime.timedelta(days=days)}) is beyond the range of the ERA5 dataset ({pd.to_datetime(ERA5_ds.valid_time.max().data).to_pydatetime()}). Please choose a shorter simulation time or provide an ERA5 dataset that covers the desired time range.")

    ############################### Heat flow model computation ##############################
    plt.ion()
    # Set up plots
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout(pad=4.5)
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_xlabel("Temperature [K]")
    ax[0].grid(visible=True)

    # ax[1].plot(Q_solar_arr.index, Q_solar_arr.ghi.values)
    # Plot solar flux curve for context
    ssrd = ERA5_ds.ssrd/3600 # Divide by 3600 to convert from J/m^2 to W/m^2

    if cmd_inputs.repeat == "day":
        ssrd.sel(valid_time=slice(start_dt,start_dt+datetime.timedelta(days=1))).plot(ax=ax[1])
    elif cmd_inputs.repeat == "year":
        ssrd.sel(valid_time=slice(start_dt, start_dt + datetime.timedelta(days=365))).plot(ax=ax[1])
    else:
        ssrd.sel(valid_time=slice(start_dt,start_dt+datetime.timedelta(days=days))).plot(ax=ax[1])
    # ax[1].set_xlabel("Time [days]")
    ax[1].set_ylabel("Solar flux [w/m^2]")
    ax[1].tick_params(axis='x',rotation=45)
    plt.ion()

    images = []
    T_surf = []
    Q_in = []
    Q_out = []
    count = 0
    # Loop through time
    for ti, ts in enumerate(tqdm(nt)):
        # Separate counter for outputs
        count += 1

        T_old = T.copy()
    
        # Get closest ERA5 data point to current timestep.
        ERA5_now = get_ERA5_at_ts(ERA5_ds, ts, start_dt,cmd_inputs.repeat)
        # Calculate radiative heat flux at surface of old temperature profile
        Q_rad = calculate_radiative_heat_flux(ERA5_now.t2m.data, T_old[0])
        # Calculate convective heat flux assuming free convection only (no wind)
        Q_conv = calculate_convective_heat_flux(ERA5_now.t2m.data, T_old[0], h=25)
        # Solar flux is downwelling shortwave radiation at surface from ERA5 dataset. Convert to W/m^2 from integrated J/m^2 by dividing by number of seconds in an hour
        Q_solar = ERA5_now.ssrd.data/3600

        ################ UPPER BOUNDARY #########################
        # Find conductivity and diffusivity at surface (From old temp) for ghost node calc
        if cmd_inputs.Earth:
            conductivity = calc_conductivity_earth(T_old[0])
            K_ghost = calc_K(T_old[0], vacuum=False)
        elif cmd_inputs.Moon:
            conductivity = calc_conductivity_lunar(T_old[0])
            if regolith_bool[0] == 0:
                K_ghost = calc_K_regolith(calc_conductivity_lunar(T_old[0]), calc_cp(T_old[0]))
            else:
                K_ghost = calc_K(T_old[0], vacuum=True)

        # Calculate ghost node temperature above surface using center finite difference method
        T_ghost_top = T_old[1] - (((2 * dx) / conductivity) * (Q_rad + Q_conv - Q_solar))

        # Reinforce upper boundary condition using ghost node "above" surface
        T[0] = T_old[0] + K_ghost * dt * ((T_old[1] - (2 * T_old[0]) + T_ghost_top) / dx**2)

        ################ MAIN COLUMN #########################
        # Loop through space using finite difference explicit discretization of Diffusion Eq.
        for i in range(1, nx - 1):
            # if we're in rock:
            if regolith_bool[i] == 1:
                # If we're on Earth
                if cmd_inputs.Earth:
                    K = calc_K(T_old[i], vacuum=False)
                # If we're on the Moon
                elif cmd_inputs.Moon:
                    K = calc_K(T_old[i], vacuum=True)

            # If we're in regolith (will only happen on the moon)
            elif regolith_bool[i] == 0:
                K = calc_K_regolith(
                    calc_conductivity_lunar(T_old[i]), calc_cp(T_old[i])
                )

            # Double check we're not violating maximum timestep condition
            dtmax = calc_dtmax(dx, K)
            if dt >= dtmax:
                warnings.warn("timestep: " + str(dt) + " is greater than " + str(dtmax))

            # forward explicit finite difference approximation (the model!)
            T[i] = T_old[i] + K * dt * (
                (T_old[i + 1] - (2 * T_old[i]) + T_old[i - 1]) / dx**2
            )

        ################ LOWER BOUNDARY #########################
        if cmd_inputs.lower_boundary_type == "insulated":
            # Find diffusivity at base for ghost node calc
            if cmd_inputs.Earth:
                K = calc_K(T_old[-1], vacuum=False)
            elif cmd_inputs.Moon:
                if regolith_bool[-1] == 0:
                    K = calc_K_regolith(
                        calc_conductivity_lunar(T_old[-1]), calc_cp(T_old[-1])
                    )
                else:
                    K = calc_K(T_old[-1], vacuum=True)

            # Calculate ghost node temp above lower boundary
            T_ghost_bottom = T_old[-2]
            # Reinforce lower boundary condition using ghost node below lower boundary
            T[-1] = T_old[-1] + K * dt * (
                (T_old[-2] - (2 * T_old[-1]) + T_ghost_bottom) / dx**2
            )
        elif cmd_inputs.lower_boundary_type == "fixed":
            T[-1] = float(cmd_inputs.T_lower)

        # Save temperature value at surface (1 pt beneath ghost node).
        T_surf.append(T[0])
        Q_in.append(Q_solar)
        Q_out.append(Q_rad)

        ################ Plotting #######################
        # Output plot every plot_interval timesteps
        if count == int(cmd_inputs.plot_interval) or ti == 0:

            # Plot temperature curve
            ax[0].plot(T, x, "-bo")
            ax[0].invert_yaxis()

            ax[0].set_xlabel("Temperature [K]")
            ax[0].set_ylabel("Depth [m]")
            ax[0].minorticks_on()

            ax[0].set_title("Surf Temp=" + str(round(T[1], 3)) + " K")
            sec_ax = ax[0].secondary_xaxis(
                "top",
                functions=(
                    lambda x: (x - 273.15) * (9 / 5) + 32,
                    lambda x: (x - 32) * (5 / 9) + 273.15,
                ),
            )
            sec_ax.set_xlabel("Temperature [°F]")
            ax[0].set_xlim(180, 350)

            # Plot dotted grey line to show basalt/regolith transition. Will not plot in Earth case
            if cmd_inputs.Moon:
                ax[0].axhline(
                    y=x[np.where(np.roll(regolith_bool, 1) != regolith_bool)[0][-1]],
                    linestyle=":",
                    color="grey",
                    alpha=0.5,
                )

            # Plot solar forcing curve on second plot
            vlines = ax[1].vlines(ERA5_now.valid_time, 0, 1050, colors="r")
            # ax[1].minorticks_on()

            # Set limits
            # ax[1].set_xlim(0, total_time / 60 / 60 / 24 / 365)
            ax[1].set_ylim(-5, 1200)
            ax[1].set_title(
                "Time Elapsed=" + str(round(ts / 60 / 60 / 24, 3)) + " days"
            )
            # plt.show()
            # Manually draw and flush events
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            # Reset counter
            count = 0
            # Save current plot image to array
            img_buf = io.BytesIO()
            fig.savefig(img_buf)
            images.append(imageio.imread(img_buf))
            vlines.remove()
        # Reset axis 0. Add back grid
        ax[0].cla()

        ax[0].grid(visible=True)

    # Once timestepping loop is done, write out results

    # Cerate directory based off current time
    out_dir = 'output/'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.mkdir(out_dir)

    # Convert cmd args to dictionary, then to dataframe
    args_dict = vars(cmd_inputs)
    df_args = pd.DataFrame({key: [value] for key, value in args_dict.items()})
    # Output as csv
    df_args.to_csv(out_dir + '/run_parameters.csv')

    # Save gif
    imageio.mimsave(out_dir + "/animation.gif", images)

    # Create dictionary of surface temp values and full time array
    out_dict = {"time": nt, "surf_temp": T_surf, "Q_out": Q_out, "Q_in": Q_in}
    # Convert to Dataframe
    df = pd.DataFrame(out_dict)

    plt.figure()
    plt.plot(nt/60/60/24,np.array(Q_in)-np.array(Q_out),label='net')
    plt.plot(nt/60/60/24,Q_out,label='outgoing')
    plt.plot(nt/60/60/24,Q_in,label='ingoing')
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel('Time [days]')
    plt.ylabel('Heat Flux [W/m²]')

    plt.figure()
    plt.plot(nt/60/60/24,T_surf)
    plt.xlabel('Time [days]')
    plt.ylabel('Surface Temperature [K]')
    plt.grid(visible=True)
    # plt.savefig(out_dir + '/heat_flux.png')
    # Output as csv
    df.to_csv(out_dir + "/surf_temp.csv")

    # Print out surface temp statistics
    print("Max Surface Temp: " + str(np.max(T_surf)) + " K")
    print("Min Surface Temp: " + str(np.min(T_surf)) + " K")    

if __name__ == "__main__":
    main()
