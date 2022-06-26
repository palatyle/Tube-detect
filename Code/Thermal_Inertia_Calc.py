from os import read
import numpy as np
from pysolar import solar
from datetime import datetime
import pytz
from osgeo import gdal
import matplotlib.pyplot as plt

import data_wrangle 

#data_wrangle.get_dt_obj(dt)

'''
#put this into own function -- Still need to do but awaiting instruction

    # Define time zone and datetime of flight
    tzone = 'US/Mountain'

    # Create datetime object
    flight_dt = datetime.fromisoformat(args.date)

    # Get timezone aware datetime object
    flight_dt_tz_aware = data_wrangle.get_dt_obj(flight_dt, tzone)

    # Get solar azimuth and altitude
    s_alt,s_azi = data_wrangle.get_solar_azi_alt(flight_dt_tz_aware,tube_lat,tube_lon)
'''

raster_day,raster_GDAL_day,_ = data_wrangle.read_in_raster("E:\\Downloaded_data\\needed_files\\day_ortho_16bit_resample_clip.tif")
#we will also be reading in night data here
raster_night,raster_GDAL_night,_ = data_wrangle.read_in_raster("E:\\Downloaded_data\\needed_files\\night_ortho.tif")


'''pos = plt.imshow(day_temp)
plt.colorbar(pos)
plt.show()
print("wow!")
'''


angular_frequency = 7.27*10**(-5) # rad/sec                maltese et al
day_sat_time = 3960000 # 1100*60*60         Scheidt et al
night_sat_time = 7956000 # 2210*60*60       Scheidt et al
day_temp = raster_day[5,:,:]
night_temp = raster_night
t_max = 5040000 # 1400*60*60 -> hours to seconds conversion      scheidt et al
t_min = 720000 # 0200*60*60 -> hours to seconds conversion       scheidt et al

Tmax = (day_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_max)) -
(np.cos(angular_frequency*day_sat_time))])/((np.cos(angular_frequency*day_sat_time)) -(np.cos(angular_frequency*night_sat_time))))

Tmin = (night_temp +((day_temp - night_temp)*[(np.cos(angular_frequency*t_min)) -
(np.cos(angular_frequency*night_sat_time))])/((np.cos(angular_frequency*day_sat_time))-(np.cos(angular_frequency*night_sat_time))))


temp_change = (Tmax - Tmin)


albedo,albedo_GDAL,_ = data_wrangle.read_in_raster("E:\\Data\\Georeference_Outputs\\Average.tiff")
ATI = ((1-albedo)/temp_change)

b = ((np.tan(angular_frequency*t_max))/(1-(np.tan(angular_frequency*t_max))))
solar_constant = 1367 # W/m**2 for earth        V. M. Fedorov 
Ct_transmittance = 0.75 # atmospheric transmittance for Earth     scheidt et al  
flight_dt = datetime.fromisoformat('2022-06-17 13:00:00')# Get timezone aware datetime object
flight_dt_tz_aware = data_wrangle.get_dt_obj(flight_dt, 'US/Mountain')
solar_declination = data_wrangle.calc_sol_dec(flight_dt_tz_aware)



phase_diff_1 = (np.arctan(b/(1+b)))     #scheidt et al
phase_diff_2 = (np.arctan((b*np.sqrt(2))/(1+(b*np.sqrt(2)))))   #scheidt et al

latitude = 43.4956
xi_constant = (np.arccos(np.tan(solar_declination)*np.tan(np.deg2rad(latitude))))    #call deg2rad around all latitudes!


A1_fourier = (((2/np.pi)*(np.sin(solar_declination)*(np.sin(np.deg2rad(latitude))))) + (
    (1/2*np.pi)*(np.cos(solar_declination)*(np.cos(np.deg2rad(latitude))))) * ([np.sin(2*xi_constant) + (2*xi_constant)]))


A2_fourier = ((((2*np.sin(solar_declination)*(np.sin(np.deg2rad(latitude))))/(2*np.pi))*(np.sin(2*xi_constant))) + (
    (2*np.cos(solar_declination)*(np.cos(np.deg2rad(latitude)))/(np.pi*(2**2 - 1)))*[(2*(np.sin(2*xi_constant))*(np.cos(xi_constant))) - 
    ((np.cos(2*xi_constant))*(np.sin(xi_constant)))]))

thermal_inertia = ((ATI*((solar_constant*Ct_transmittance)/np.sqrt(angular_frequency)))*(
    ((A1_fourier*((np.cos((angular_frequency*night_sat_time)-phase_diff_1))-(np.cos((angular_frequency*day_sat_time)-phase_diff_1))))/
    (np.sqrt(1+(1/b)+(1/(2*b**2))))) +
    ((A2_fourier+((np.cos((angular_frequency*night_sat_time)-phase_diff_2))-(np.cos((angular_frequency*day_sat_time)-phase_diff_2))))/
    (np.sqrt(2+(np.sqrt(2)/b)+(1/(2*b**2)))))))
