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


angular_frequency = 7.2921159*10**(-5) # rad/sec             add calculations into here
day_sat_time = # insert input to the function--overpass time
night_sat_time = # insert input to the function-- overpass time
day_temp = # temperature at time day_sat_time
night_temp = # temperature at time night_sat_time
t_max = 1400 #??? Ask Tyler and find out because t_max and t_min (with the small t's) are not defined in the paper.
t_min = 0200 #datetime objects

Tmax = (day_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_max)) -
 (np.cos(angular_frequency*day_sat_time))])/((np.cos(angular_frequency*day_sat_time)) -(np.cos(angular_frequency*night_sat_time))))

Tmin = (night_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_min)) -
 (np.cos(angular_frequency*night_sat_time))])/((np.cos(angular_frequency*day_sat_time))-(np.cos(angular_frequency*night_sat_time))))


temp_change = (Tmax - Tmin)


albedo = average #need to get this from Albedo_Stats or my files and properly format it with Tyler's help
ATI = ((1-albedo)/temp_change)

b = ((np.tan(angular_frequency*t_max))/(1-(np.tan(angular_frequency*t_max))))
solar_constant = 1380 # w/m^2 for earth
Ct_transmittance = 0.75 # atmospheric transmittance for earth. 0.75 obtained from email.
solar_declination = #add equation here (function) gonna get from tyler

phase_diff_1 = #get from sheidt paper
phase_diff_2 = #get from sheidt paper
#change these in all the code? (thermal inertia calculation)


latitude = #insert latitude for HHA input
xi_constant = (np.arccos(np.tan(solar_declination)*np.tan(latitude)))    #call deg2rad around all latituces!


A1_fourier = (((2/np.pi)*(np.sin(solar_declination)*(np.sin(latitude)))) + (
    (1/2*np.pi)*(np.cos(solar_declination)*(np.cos(latitude)))) * ([np.sin(2*xi_constant) + (2*xi_constant)]))


A2_fourier = ((((2*np.sin(solar_declination)*(np.sin(latitude)))/(2*np.pi))*(np.sin(2*xi_constant))) + (
    (2*np.cos(solar_declination)*(np.cos(latitude))/(np.pi*(2**2 - 1)))*[(2*(np.sin(2*xi_constant))*(np.cos(xi_constant))) - 
    ((np.cos(2*xi_constant))*(np.sin(xi_constant)))]))

thermal_inertia =



