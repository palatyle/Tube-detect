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