import sys
import io
import warnings
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import imageio

def calc_K(temp,vacuum=False):
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
        A = 3.59e-3 # (cm^2/s) 
        B = 1.65 # (cm^2 K/s)
        C = -0.68e-12 #(cm^2/s K^3)
    elif vacuum == True:
        A = 3.03e-3 # (cm^2/s) 
        B = 1.30 # (cm^2 K/s)
        C = 1.93e-12 #(cm^2/s K^3)
    therm_diffusivity = A + (B/temp) + (C * temp**3) # cm^2/s 
    # print(therm_diffusivity/10000)
    return therm_diffusivity/10000

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
    return conductivity/(density*cp)

def calc_conductivity(T,Kc=3.4e-3,chi=2.7):
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
    return Kc * (1 + chi * (T/350) ** 3)
    
def calc_cp(T,c0=-3.6125,c1=2.7431,c2=2.3616e-3,c3=-1.234e-5,c4=8.9093e-9):
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


def calc_cos_eq(temps,times):
    """Calculate parameters for equation of cosine function with the form of:
    y = A*cos(pi/p * (x-h)) + k

    Parameters
    ----------
    temps : list
        list with length of 2. 0th element in max temp, 1st element is min temp
    times : list
        list with length of 2. 0th element in max time, 1st element is min time

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
    A_func = abs((temps[0]-temps[1])/2)
    k_func = (temps[0]+temps[1])/2
    p_func = times[0]-times[1]
    h_func = times[1]
    return A_func, k_func, p_func, h_func

def get_penetration_depth(temp_arr,bulk_temp,threshold=1):
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
    diff = temp_arr-bulk_temp

    
    # Find all indices where above difference is greater than threshold
    penetration_depths = np.where(diff>=threshold)
    
    #Convert 1st element (the only element) to list
    pdep_list = penetration_depths[0].tolist()
    
    if not(pdep_list):
        return 0
    else:
        # Get last value since we care about the deepest change
        return pdep_list[-1]


def find_reg_depth_idx(roof_thick,dom_len,dom_arr):
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
        reg_bool[len(dom_arr)-round(len(dom_arr)/(dom_len/roof_thick))::-1] = 0

        return reg_bool




def calc_dtmax(dx_step,therm_diff):
    return (dx_step**2)/(2*therm_diff)

Earth_flag = True

save_gif = True

if Earth_flag == True:
    regolith_thickness = 0
    tube_roof = 5
elif Earth_flag == False:
    # Model domain length - Rough size of tube roof 
    regolith_thickness = 0.5 # (m) 5m - (Fa and Wieczorek, 2012)
    tube_roof = 5 # (m)
    
L = regolith_thickness + tube_roof # 5m basalt, 5m regolith (m)

# Timestep divisions
timesteps = 150


# Number of grid points
nx = 100


if Earth_flag == True:
    # Seconds in 1 day
    day = 24*60*60*365
elif Earth_flag == False:
    # Seconds in 1 lunar day. lunar day  29.53 earth days
    day = 65*365*24*60*60

#  Space array + grid spacing
x,dx = np.linspace(0,L,nx,retstep=True)

regolith_bool = find_reg_depth_idx(tube_roof,L,x)

if Earth_flag == True:
    # Max and min diurnal temps/times from UAS data (median from day and night). 
    max_temp = 42.37+273.15 # (K)
    max_temp_time = 14 * 60 * 60 # (seconds since midnight)
    min_temp = 12.67+273.15 # (K)
    min_temp_time = 2 * 60 * 60 # (seconds since midnight)
    
    # Bulk rock temperature. 
    T_basalt = (max_temp+min_temp)/2 # (K) 
elif Earth_flag == False:
    # Max and min diurnal temps/times from Diviner data. 
    max_temp = 356.43 # (K) Williams et al. 2017 over Highland 1
    max_temp_time = (29.53/2) * 24 * 60 * 60 # (seconds since midnight) https://svs.gsfc.nasa.gov/12739
    min_temp = 87.76 # (K) Williams et al. 2017 over Highland 1
    min_temp_time = 0  # (seconds since midnight)

    # Bulk rock temperature. 
    T_basalt = (max_temp+min_temp)/2 # (K) 
    # T_basalt = min_temp
# Get cosine equation parameters 
A,k,p,h = calc_cos_eq([max_temp,min_temp],[max_temp_time,min_temp_time])


lower_boundary = 273 # 290 # 45 # K Paige 2010



dt_range = np.logspace(2,9,1000)
dt_bool=[]
print('Finding optimal dt value')
for dt in tqdm(dt_range):
    T_list = np.linspace(lower_boundary,max_temp,100000)

    if Earth_flag == True:
        K_rock_no_vac = calc_K(T_list,vacuum=False)
        dtmax_rock_no_vac = calc_dtmax(dx,K_rock_no_vac)
        bad_rock_no_vac = np.where(dt <= dtmax_rock_no_vac)
        dt_bool.append(not(bad_rock_no_vac[0].tolist()))
        # print(not(bad_rock[0].tolist()))
    elif Earth_flag == False:
        K_rock = calc_K(T_list,vacuum=True)
        K_reg = calc_K_regolith(calc_conductivity(T_list),calc_cp(T_list))
        dtmax_rock = calc_dtmax(dx,K_rock)
        bad_rock = np.where(dt <= dtmax_rock)
        dtmax_reg = calc_dtmax(dx,K_reg)
        bad_reg = np.where(dt >= dtmax_reg)
        dt_bool.append(not(bad_reg[0].tolist()) and not(bad_rock_no_vac[0].tolist()))

# Find where false flips to true minus 10 as a buffer (since we're not testing every possible #)
best_dt_idx = np.where(dt_bool)[0][0] - 10

# Time array + timestep
dt = dt_range[best_dt_idx]
nt = np.arange(1,day,dt)
print('dt='+str(dt))
print('dx='+str(dx))

# Temperature curve 
temp_curve = -A*np.cos((np.pi/p)*(nt-h)) + k

# Create temperature array. Set all values equal to basalt temp
T = np.ones_like(x)*T_basalt

# Boundary condition is set by temperature curve
T[0] = temp_curve[0] # (K) 


fig, ax = plt.subplots(1,2)
fig.tight_layout(pad=1.8)
ax[0].set_ylabel('Depth [m]')
ax[0].set_xlabel('Temperature [K]')
ax[0].grid(visible=True)
ax[1].set_ylabel('Temperature [K]')
if Earth_flag == True:
    ax[1].set_xlabel('Time (hours)')
elif Earth_flag == False:
    ax[1].set_xlabel('Time [years]')

images = []
penetration_depth=[]
count = 0 
for ti,ts in enumerate(tqdm(nt)):
    count += 1
    # Setup new temperature array for future time. Fill with zeros
    Tnew = np.zeros_like(T)
    # Loop through space using finite difference explicit discretization of Diffusion Eq.
    for i in range(1,nx-1):
        if regolith_bool[i] == 1:
            if Earth_flag == True:
                K = calc_K(T[i],vacuum=False)
            elif Earth_flag == False:
                K = calc_K(T[i],vacuum=True)
        elif regolith_bool[i] == 0:
            K = calc_K_regolith(calc_conductivity(T[i]),calc_cp(T[i]))
            
        dtmax =(dx**2)/(2*K)
        if dt >= dtmax:
            warnings.warn('timestep: '+str(dt)+' is greater than ' + str(dtmax))
        
        # forward explicit finite difference approximation 
        Tnew[i] = T[i] + K*dt*((T[i+1]-(2*T[i])+T[i-1])/dx**2)
        
    # Reinforce boundary conditions
    Tnew[0] = temp_curve[ti]
    # Tnew[-1] = T[-1]
    Tnew[-1] = lower_boundary
    
    # Set current T equal to Tnew
    T = Tnew
    

    
    # penetration_depth.append(get_penetration_depth(T,T_basalt))
    

    


    if save_gif == True and count == 250 or ti == 0:
        
        # Plot temperature curve
        ax[0].plot(Tnew,x,'-bo')

        ax[0].text(275,1,str(round(T[1],3)))
        # ax[0].axhline(y=x[penetration_depth[ti]],color='r')

        try:
            ax[0].axhline(y=x[np.where(np.roll(regolith_bool,1)!=regolith_bool)[0][-1]],linestyle=":",color="grey",alpha=0.5)
        except:
            None

        # Plot solar forcing curve 
        if Earth_flag == True:
            ax[1].plot(ts/60/60,temp_curve[ti],'bo')
        elif Earth_flag == False:
            ax[1].plot(ts/60/60/24/365,temp_curve[ti],'bo')
        ax[0].invert_yaxis()
        
        # Set limits
        if Earth_flag == True:
            ax[1].set_xlim(0,day/60/60)  
        elif Earth_flag == False:
            ax[1].set_xlim(0,day/60/60/24/365)

        if lower_boundary < min_temp:
            ax[0].set_xlim(273.15-5,max_temp)
        elif lower_boundary >= min_temp:
            ax[0].set_xlim(273.15-5,max_temp)
            
        ax[1].set_ylim(min_temp,max_temp)
    
        ax[0].set_ylabel('Depth [m]')
        ax[0].set_xlabel('Temperature [K]')
        
        ax[0].minorticks_on()
        ax[1].minorticks_on()

        # plt.pause(.00001)
        count = 0
        img_buf = io.BytesIO()
        fig.savefig(img_buf)
        images.append(imageio.imread(img_buf))

    ax[0].cla()
    ax[0].grid(visible=True)


    
if save_gif == True:
    if Earth_flag == True:
        imageio.mimsave('Earth_temp_p_depth-'+str(round(max(x[penetration_depth]),2))+'m.gif',images)
    elif Earth_flag == False:
        imageio.mimsave('lunar_temp_p_depth-'+str(round(max(x[penetration_depth]),2))+'m_reg_depth-'+str(regolith_thickness)+'m.gif',images)

print(str(round(max(x[penetration_depth]),2)))