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

#put this into own function

    # Define time zone and datetime of flight
    tzone = 'US/Mountain'

    # Create datetime object
    flight_dt = datetime.fromisoformat(args.date)

    # Get timezone aware datetime object
    flight_dt_tz_aware = get_dt_obj(flight_dt, tzone)

    # Get solar azimuth and altitude
    s_alt,s_azi = get_solar_azi_alt(flight_dt_tz_aware,tube_lat,tube_lon)


angular_frequency = 7.2921159*10^(-5)
day_sat_time = # insert input to the function--overpass time
night_sat_time = # insert input to the function-- overpass time
day_temp = # temperature at time day_sat_time
night_temp = # temperature at time night_sat_time
t_max = #??? Ask Tyler and find out because t_max and t_min (with the small t's) are not defined in the paper.
t_min = #?

Tmax = day_sat_time +((day_temp - night_temp)*[(np.cos(angular_frequency*t_max)) -
 (np.cos(angular_frequency*day_sat_time))])/((np.cos(angular_frequency*day_sat_time)) -(np.cos(angular_frequency*night_sat_time)))

Tmin = night_sat_time+((day_temp - night_temp)*[(np.cos(angular_frequency*t_min)) -
 (np.cos(angular_frequency*night_sat_time))])/((np.cos(angular_frequency*day_sat_time))-(np.cos(angular_frequency*night_sat_time)))


temp_change = abs(Tmax -Tmin)


albedo = average #need to get this from Albedo_Stats or my files and properly format it with Tyler's help
ATI = (1-albedo)/temp_change



thermal_inertia = 



