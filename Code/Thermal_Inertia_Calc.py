import numpy as np
from pysolar import solar
from datetime import datetime
import pytz


def get_dt_obj(dt,tzone = 'US/Mountain'):
    '''
    Returns a timezone aware datetime object
    
    Parameters
    ----------
    dt: datetime object
    tzone: time zone string
    Returns
    -------
    Output localized timezone from pytz package
    '''
    tz = pytz.timezone(tzone)
    return tz.localize(dt,is_dst=True)



def get_solar_azi_alt(dt,lat,lon):
    '''
    Returns solar altitude and azimuth using solarpy packages
    Parameters
    ----------
    dt: Timezone aware datetime object
    lat: latitude of general tube location [deg]
    lon: longitude of general tube location [deg]
    Returns
    -------
    alt: altitude of sun from tangential plane of Earth's surface at lat,lon coords 0-90 [deg]
    azi: azimuth of sun from north. 0 to 360 [deg]
    '''
    azi = solar.get_azimuth(lat,lon,dt)
    alt = solar.get_altitude(lat,lon,dt)
    return alt, azi

#put this into own function -- Still need to do but awaiting instruction

    # Define time zone and datetime of flight
    tzone = 'US/Mountain'

    # Create datetime object
    flight_dt = datetime.fromisoformat(args.date)

    # Get timezone aware datetime object
    flight_dt_tz_aware = get_dt_obj(flight_dt, tzone)

    # Get solar azimuth and altitude
    s_alt,s_azi = get_solar_azi_alt(flight_dt_tz_aware,tube_lat,tube_lon)

def deg2rad(angle):
    """_summary_

    Args:
        angle (float): angle in degrees

    Returns:
        float: angle in radians
          """
    return angle*((2*np.pi)/360)


angular_frequency = 7.27*10**(-5) # rad/sec                maltese et al
day_sat_time = 3960000 # 1100*60*60         Scheidt et al
night_sat_time = 7956000 # 2210*60*60       Scheidt et al
day_temp = # temperature at time day_sat_time
night_temp = # temperature at time night_sat_time
t_max = 5040000 # 1400*60*60 -> hours to seconds conversion      scheidt et al
t_min = 720000 # 0200*60*60 -> hours to seconds conversion       scheidt et al

Tmax = (day_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_max)) -
 (np.cos(angular_frequency*day_sat_time))])/((np.cos(angular_frequency*day_sat_time)) -(np.cos(angular_frequency*night_sat_time))))

Tmin = (night_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_min)) -
 (np.cos(angular_frequency*night_sat_time))])/((np.cos(angular_frequency*day_sat_time))-(np.cos(angular_frequency*night_sat_time))))


temp_change = (Tmax - Tmin)


albedo = average #need to get this from Albedo_Stats or my files and properly format it with Tyler's help
ATI = ((1-albedo)/temp_change)

b = ((np.tan(angular_frequency*t_max))/(1-(np.tan(angular_frequency*t_max))))
solar_constant = 1367 # W/m**2 for earth        V. M. Fedorov 
Ct_transmittance = 0.75 # atmospheric transmittance for Earth     scheidt et al  
solar_declination = #add equation here (function) gonna get from tyler

phase_diff_1 = (np.arctan(b/(1+b)))     #scheidt et al
phase_diff_2 = (np.arctan((b*np.sqrt(2))/(1+(b*np.sqrt(2)))))   #scheidt et al
#change these in all the code? (thermal inertia calculation) ---> WHAT DOES THIS MEEEAN?


latitude = #insert latitude for HHA input
xi_constant = (np.arccos(np.tan(solar_declination)*np.tan(deg2rad(latitude))))    #call deg2rad around all latitudes!


A1_fourier = (((2/np.pi)*(np.sin(solar_declination)*(np.sin(deg2rad(latitude))))) + (
    (1/2*np.pi)*(np.cos(solar_declination)*(np.cos(deg2rad(latitude))))) * ([np.sin(2*xi_constant) + (2*xi_constant)]))


A2_fourier = ((((2*np.sin(solar_declination)*(np.sin(deg2rad(latitude))))/(2*np.pi))*(np.sin(2*xi_constant))) + (
    (2*np.cos(solar_declination)*(np.cos(deg2rad(latitude)))/(np.pi*(2**2 - 1)))*[(2*(np.sin(2*xi_constant))*(np.cos(xi_constant))) - 
    ((np.cos(2*xi_constant))*(np.sin(xi_constant)))]))

thermal_inertia = ((ATI*((solar_constant*Ct_transmittance)/np.sqrt(angular_frequency)))*(
    ((A1_fourier*((np.cos((angular_frequency*night_sat_time)-phase_diff_1))-(np.cos((angular_frequency*day_sat_time)-phase_diff_1))))/
    (np.sqrt(1+(1/b)+(1/(2*b**2))))) +
    ((A2_fourier+((np.cos((angular_frequency*night_sat_time)-phase_diff_2))-(np.cos((angular_frequency*day_sat_time)-phase_diff_2))))/
    (np.sqrt(2+(np.sqrt(2)/b)+(1/(2*b**2)))))))


