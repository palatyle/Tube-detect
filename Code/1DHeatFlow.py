import sys
import io

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

def get_penetration_depth(temp_arr,threshold=1):
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
    # Find difference between each value and its neighbor
    # Add 0 to bottom to maintain length
    diff = abs(np.diff(temp_arr,prepend=0))

    # Find all indices where above difference is greater than theshold
    penetration_depths = np.where(diff>=threshold)
    
    # Get last value since we care about the deepest change
    return penetration_depths[0][-1]

# Model domain length - Rough size of tube roof 
L = 5 # (m)
# Timestep divisions
timesteps = 500

save_gif = True


# Number of grid points
nx = 50 
# Seconds in 1 day
day = 24*60*60
# Time array + timestep
nt,dt = np.linspace(1,day,timesteps,retstep=True) 
#  Space array + grid spacing
x,dx = np.linspace(0,L,nx,retstep=True)


# Bulk rock temperature. 
T_basalt = 12+273.15 # (K) 

# Max and min diurnal temps/times from UAS data (median from day and night). 
max_temp = 42.37+273.15 # (K)
max_temp_time = 14 * 60 * 60 # (seconds since midnight)
min_temp = 12.67+273.15 # (K)
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
T[0] = temp_curve[0] # (K) 


plt.ion()
fig, ax = plt.subplots(1,2)
ax[0].set_ylabel('Depth [m]')
ax[0].set_xlabel('Temperature [K]')
ax[0].invert_yaxis()

ax[1].set_ylabel('Temperature [K]')
ax[1].set_xlabel('Time (hours)')

images = []
penetration_depth=[]
for ti,ts in enumerate(nt):
    # Setup new temperature array for future time. Fill with zeros
    Tnew = np.zeros_like(T)
    
    # Loop through space using finite difference explicit discretization of Diffusion Eq.
    for i in range(1,nx-1):
        K = calc_K(T[i])
        dtmax =(dx**2)/(2*K)
        if dt >= dtmax:
            sys.exit('timestep: '+str(dt)+' is greater than ' + str(dtmax))
        Tnew[i] = T[i] + K*dt*((T[i+1]-(2*T[i])+T[i-1])/dx**2)
        
    # Reinforce boundary conditions
    Tnew[0] = temp_curve[ti]
    Tnew[-1] = T[-1]
    
    # Set current T equal to Tnew
    T = Tnew


    # Plot temperature curve
    ax[0].plot(Tnew,x)
    
    penetration_depth.append(get_penetration_depth(T))
    ax[0].axhline(y=x[penetration_depth[ti]],color='r')
    # Plot solar forcing curve 
    ax[1].plot(ts/60/60,temp_curve[ti],'bo')
    ax[0].invert_yaxis()
    # Set limits
    ax[0].set_xlim(min_temp-5,max_temp)
    ax[1].set_xlim(0,day/60/60)
    ax[1].set_ylim(min_temp,max_temp)

    ax[0].set_ylabel('Depth [m]')
    ax[0].set_xlabel('Temperature [K]')
    
    ax[0].minorticks_on()
    ax[1].minorticks_on()

    plt.pause(.001)

    if save_gif == True:
        img_buf = io.BytesIO()
        fn = 'lunar'+str(round(ts))+'.png'
        fig.savefig(img_buf)
        images.append(imageio.imread(img_buf))
    ax[0].cla()

if save_gif == True:
    imageio.mimsave('Earth_temp_p_depth-'+str(round(max(x[penetration_depth]),2))+'m.gif',images)