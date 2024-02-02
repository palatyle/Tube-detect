'''
1D heat flow model. Can calculate heat flow for basalt on the Moon or Earth. On Moon, a two layer model (regolith and basalt) 
is used. The model will automatically find the largest possible timestep that satisfies numerical stability conditions. 

Author: Tyler Paladino

Run from command line. 
Ex for Earth: 
python lava_tube_heat_flow.py --Earth --roof_thickness 5 --regolith_thickness 0 --lower_boundary 273

Ex. for Moon:
python lava_tube_heat_flow.py --Moon --roof_thickness 5 --regolith_thickness 5 --lower_boundary 45

Output will be a .gif and a .csv in the same directory the code was run from. csv contains data
that can be visualzied in heat_flow_viz.py

Required packages:
imageio
matplotlib
numpy
pandas
tqdm
'''

import argparse
import io
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Earth",
                        help="Set Earth to True",
                        action="store_true"
                        )
    
    parser.add_argument("--Moon",
                        help="Set Moon to True",
                        action="store_true"
                        )

    parser.add_argument("--roof_thickness", 
                        help="Set tube roof thickness in meters",
                        required = True)

    parser.add_argument("--regolith_thickness", 
                        help="Set regolith thickness in meters",
                        required = True)

    parser.add_argument("--lower_boundary", 
                        help="Set lower boundary. Either 45 K, 290 K, or T_basalt (Average of max/min temps from field)",
                        required = True)

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


def calc_conductivity(T, Kc=3.4e-3, chi=2.7):
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


def calc_cos_eq(temps, times):
    """Calculate parameters for equation of cosine function with the form of:
    y = A*cos(pi/p * (x-h)) + k

    Parameters
    ----------
    temps : list
        list with length of 2. 0th element is max temp, 1st element is min temp
    times : list
        list with length of 2. 0th element is max time, 1st element is min time

    Returns
    -------
    A_func : int
        Amplitude of Cosine
    k_func : int
        Vertial shift of Cosine
    p_func : int
        Period of Cosine
    h_func : int
        Phase shift of Cosine
    """
    # Calculating parameters for cosine eq.
    A_func = abs((temps[0] - temps[1]) / 2)
    k_func = (temps[0] + temps[1]) / 2
    p_func = times[0] - times[1]
    h_func = times[1]
    return A_func, k_func, p_func, h_func


def get_penetration_depth(temp_arr, bulk_temp, threshold=1):
    """Calculates the deepest location where temperature penetrates.
    Threshold allows one to tune how large of a change must occur between
    grid points to be deemed significant

    Parameters
    ----------
    temp_arr : arr
        Temperature array (K)
    threshold : int, optional
        Threshold temperature, by default 1 K

    Returns
    -------
    int
        Index of greatest penetration depth
    """
    # Find difference between each value and bulk rock temperature
    # Add 0 to bottom to maintain length
    # diff = abs(np.diff(temp_arr,prepend=0))
    diff = temp_arr - bulk_temp

    # Find all indices where above difference is greater than threshold
    penetration_depths = np.where(diff >= threshold)

    # Convert 1st element (the only element) to list
    pdep_list = penetration_depths[0].tolist()

    if not (pdep_list):
        return 0
    else:
        # Get last value since we care about the deepest change
        return pdep_list[-1]


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

def find_best_dt(lower_t_bound, del_x, temp_min, temp_max, args):
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
    dt_range_test = np.logspace(2, 9, 1000)
    print("Finding optimal dt value")
    # for dt in tqdm(dt_range_test):

    # Create list of temperatures based on lowest to highest temps in domain
    if lower_t_bound < temp_min:
        T_list = np.linspace(lower_t_bound, temp_max, 100000)
    else:
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
        K_reg = calc_K_regolith(calc_conductivity(T_list), calc_cp(T_list))
        
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

        
    best_dt = dt_range_test[best_dt_idx]
    return best_dt

def main():
    ############################# Define inputs ########################################
    
    cmd_inputs = parse()
    # Regolith thickness
    regolith_thickness = float(cmd_inputs.regolith_thickness) # Should be 0 for Earth
    # Tube roof thickness
    tube_roof = float(cmd_inputs.roof_thickness)

    if cmd_inputs.Earth:
        # Time to run model for
        total_time = 24 * 60 * 60 * 365

        # Max and min diurnal temps/times from UAS data (median from day and night).
        max_temp = 42.37 + 273.15  # (K)
        max_temp_time = 14 * 60 * 60  # (seconds since midnight)
        min_temp = 12.67 + 273.15  # (K)
        min_temp_time = 2 * 60 * 60  # (seconds since midnight)
    elif cmd_inputs.Moon:
        # Time to run model for
        total_time = 60 * 365 * 24 * 60 * 60

        # Max and min diurnal temps/times from Diviner data.
        max_temp = 356.43  # (K) Williams et al. 2017 over Highland 1
        max_temp_time = ((29.53 / 2) * 24 * 60 * 60)  # (seconds since midnight) https://svs.gsfc.nasa.gov/12739
        min_temp = 87.76  # (K) Williams et al. 2017 over Highland 1
        min_temp_time = 0  # (seconds since midnight)


    # Total length of domain in meters
    L = regolith_thickness + tube_roof

    # Number of grid points
    nx = 75

    #  Space array + grid spacing
    x, dx = np.linspace(0, L, nx, retstep=True)

    # Find which parts of domain are regolith
    regolith_bool = find_reg_depth_idx(tube_roof, L, x)

    # Bulk rock temperature.
    T_basalt = (max_temp + min_temp) / 2  # (K)

    # Set lower boundary of model to either the middle between max and min temperature or as boundary set in args
    if cmd_inputs.lower_boundary == "T_basalt":
        lower_boundary = T_basalt
    else:
        lower_boundary = float(cmd_inputs.lower_boundary)
        
    ######################## Compute time & temperature arrays ############################
    
    # Precompute smallest timestep value that satisfies numerical requirements
    dt = find_best_dt(lower_boundary,dx,min_temp,max_temp,cmd_inputs)

    # Create time array based on dt
    nt = np.arange(1, total_time, dt)
    print("dt=" + str(dt))
    print("dx=" + str(dx))

    # Get cosine equation parameters for upper boundary temperature forcing
    A, k, p, h = calc_cos_eq([max_temp, min_temp], [max_temp_time, min_temp_time])
    # Create temperature curve. This is what will be referenced to set the upper boundary at every timestep
    temp_curve = -A * np.cos((np.pi / p) * (nt - h)) + k

    # Create temperature array to model on. Make same length as x domain. Set all values equal to basalt temp to start
    T = np.ones_like(x) * T_basalt

    # Upper boundary condition before loop is set by 1st entry of temperature curve
    T[0] = temp_curve[0]  # (K)

    ############################### Heat flow model computation ##############################

    # Set up plots
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout(pad=1.8)
    ax[0].set_ylabel("Depth [m]")
    ax[0].set_xlabel("Temperature [K]")
    ax[0].grid(visible=True)

    ax[1].set_xlabel("Time [years]")
    ax[1].set_ylabel("Temperature [K]")


    images = []
    T_surf = []
    count = 0
    # Loop through time
    for ti, ts in enumerate(tqdm(nt)):
        # Separate counter for outputs
        count += 1
        
        # Setup new temperature array for future time. Fill with zeros
        Tnew = np.zeros_like(T)
        
        # Loop through space using finite difference explicit discretization of Diffusion Eq.
        for i in range(1, nx - 1):
            # if we're in rock:
            if regolith_bool[i] == 1:
                # If we're on Earth
                if cmd_inputs.Earth:
                    K = calc_K(T[i], vacuum=False)
                # If we're on the Moon
                elif cmd_inputs.Moon:
                    K = calc_K(T[i], vacuum=True)
                    
            # If we're in regolith (will only happen on the moon)
            elif regolith_bool[i] == 0:
                K = calc_K_regolith(calc_conductivity(T[i]), calc_cp(T[i]))

            # Double check we're not violating maximum timestep condition
            dtmax = calc_dtmax(dx, K)
            if dt >= dtmax:
                warnings.warn("timestep: " + str(dt) + " is greater than " + str(dtmax))

            # forward explicit finite difference approximation (the model!)
            Tnew[i] = T[i] + K * dt * ((T[i + 1] - (2 * T[i]) + T[i - 1]) / dx**2)

        # Once all temperatures have been calcualted at every location, reinforce boundary conditions
        Tnew[0] = temp_curve[ti]
        Tnew[-1] = lower_boundary

        # Set current T equal to Tnew
        T = Tnew

        # Save temperature value 1 grid point below surface.
        T_surf.append(T[1])
        
        ################ Plotting #######################
        # Output plot every 250 timesteps
        if count == 250 or ti == 0:

            # Plot temperature curve
            ax[0].plot(Tnew, x, "-bo")
            ax[0].invert_yaxis()

            ax[0].set_xlabel("Temperature [K]")
            ax[0].set_ylabel("Depth [m]")
            ax[0].minorticks_on()

            ax[0].set_title("Surf Temp=" + str(round(T[1], 3)) + " K")

            # set x limits of plot based on what variable lowest temp is in
            if lower_boundary < min_temp:
                ax[0].set_xlim(lower_boundary - 5, max_temp)
            elif lower_boundary >= min_temp:
                ax[0].set_xlim(min_temp - 5, max_temp)
            
            # Plot dotted grey line to show basalt/regolith transition. Will not plot in Earth case
            if cmd_inputs.Moon:
                ax[0].axhline(
                    y=x[np.where(np.roll(regolith_bool, 1) != regolith_bool)[0][-1]],
                    linestyle=":",
                    color="grey",
                    alpha=0.5
                    )

            # Plot solar forcing curve on second plot
            ax[1].plot(ts / 60 / 60 / 24 / 365, temp_curve[ti], "bo")
            ax[1].minorticks_on()

            # Set limits
            ax[1].set_xlim(0, total_time / 60 / 60 / 24 / 365)
            ax[1].set_ylim(min_temp, max_temp)
            ax[1].set_title("Current Time=" + str(round(ts / 60 / 60 / 24, 3)) + " days")

            # Reset counter
            count = 0
            # Save current plot image to array
            img_buf = io.BytesIO()
            fig.savefig(img_buf)
            images.append(imageio.imread(img_buf))

        # Reset axis 0. Add back grid
        ax[0].cla()
        ax[0].grid(visible=True)
    
    # Once timestepping loop is done, Create filename string based off regolith thickness 
    # and lower temperature boundary
    fn = "heat_flow_reg-" + str(regolith_thickness) + "m_lower_bound-" + str(lower_boundary)

    # Save gif
    imageio.mimsave(fn + "_K.gif", images)

    # Create dictionary of surface temp values and full time array
    out_dict = {"time": nt, "surf_temp": T_surf}
    # Convert to Dataframe
    df = pd.DataFrame(out_dict)
    # Output as csv
    df.to_csv(fn + "_K.csv")
    
if __name__ == '__main__':
    main()