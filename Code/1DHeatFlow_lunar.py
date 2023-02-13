import sys

import matplotlib.pyplot as plt
import numpy as np

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

def calc_K_regolith(conductivity, cp, density):
    """Calculates thermal diffusivity from conductivity, specific heat, 
    and density 

    Parameters
    ----------
    conductivity : float
        Thermal conductivity
    cp : float
        Specific heat 
    density : float
        Bulk rock density (kg/m^3)

    Returns
    -------
    float
        _description_
    """    
    return conductivity/(density*cp)

def calc_conductivity(T,Kc,chi):
    """Calculate thermal conductivity based on eq. A4 from Hayne et al. 2017

    Parameters
    ----------
    T : array
        Temperature array (K)
    Kc : float
        Phonon conductivity (W m^-1 K^-1)
    chi : float
        Radiative conductivity parameter (unitless)

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

# Model domain length - Rough size of tube roof 
L = 10 # 5m basalt, 5m regolith (m)

# Timestep divisions
timesteps = 500

# Number of grid points
nx = 30 
# Seconds in 1 lunar day
day = 14 * 24 * 60 * 60
# Time array + timestep
nt,dt = np.linspace(1,day,timesteps,retstep=True) 
#  Space array + grid spacing
x,dx = np.linspace(0,L,nx,retstep=True)


# Bulk rock temperature. 
T_basalt = 12 # (C) 

# Max and min diurnal temps/times from UAS data (median from day and night). 
max_temp = 42.37 # (C)
max_temp_time = 14 * 60 * 60 # (seconds since midnight)
min_temp = 12.67 # (C)
min_temp_time = 2 * 60 * 60 # (seconds since midnight)


# Get cosine equation parameters 
A,k,p,h = calc_cos_eq([max_temp,min_temp],[max_temp_time,min_temp_time])

# Temperature curve calculated from UAS data. 
temp_curve = -A*np.cos((np.pi/p)*(nt-h)) + k

# Calculate diffusivity for every temperature value. 
K_matrix = calc_K(temp_curve)

# Create temperature array. Set all values equal to basalt temp
T = np.ones_like(x)*T_basalt
# Boundary condition is set by temperature curve
T[0] = temp_curve[0] # (C) 

# Check explicity FTCS stabiltiy. dt must be less than dx^2/2*K
dtmax =(dx**2)/(2*K_matrix)
if dt >= min(dtmax):
    sys.exit('timestep: '+str(dt)+' is greater than ' + str(dtmax))


plt.ion()
fig, ax = plt.subplots(1,2)
ax[0].set_ylabel('distance [m]')
ax[0].set_xlabel('Temperature [C]')
ax[0].invert_yaxis()

ax[1].set_ylabel('Temperature [C]')
ax[1].set_xlabel('Time (hours)')


for ti,ts in enumerate(nt):
    # Setup new temperature array for future time. Fill with zeros
    Tnew = np.zeros_like(T)
    
    # Loop through space using finite difference explicit discretization of Diffusion Eq.
    for i in range(1,nx-1):
        Tnew[i] = T[i] + K_matrix[ti]*dt*((T[i+1]-(2*T[i])+T[i-1])/dx**2)
        
    # Reinforce boundary conditions
    Tnew[0] = temp_curve[ti]
    Tnew[-1] = T[-1]
    
    # Set current T equal to Tnew
    T = Tnew

    # Plot temperature curve
    ax[0].plot(Tnew,x)
    # Plot solar forcing curve 
    ax[1].plot(ts/60/60,temp_curve[ti],'bo')
    ax[0].invert_yaxis()
    # Set limits
    ax[0].set_xlim(min_temp-5,max_temp)
    ax[1].set_xlim(0,day/60/60)
    ax[1].set_ylim(min_temp,max_temp)
    plt.pause(.1)
    ax[0].cla()
    ax[0].set_ylabel('distance [m]')
    ax[0].set_xlabel('Temperature [C]')
    
    ax[1].set_ylabel('Temperature [C]')
    ax[1].set_xlabel('Time (hours)')
