from os import read
import numpy as np
from pysolar import solar
from datetime import datetime
import pytz
from osgeo import gdal
import operator
import matplotlib.pyplot as plt
import data_wrangle as dw
# import argparse

'''
parser2 = argparse.ArgumentParser()
parser2.add_argument("--day_thermal", help = "Full path to daytime thermal data.")
args2 = parser2.parse_args()
args2.day_thermal
print(args2.day_thermal)
'''

input_day_thermal = "E:\\Data\\Georeference_Outputs\\Day_Clip.tif"
input_night_thermal = "E:\\Data\\Georeference_Outputs\\Night_Clip.tif"
input_albedo = "E:\\Data\\Georeference_Outputs\\Near_Clip.tif"

raster_day,raster_GDAL_day,_= dw.read_in_raster(input_day_thermal)
raster_night,raster_GDAL_night,_ = dw.read_in_raster(input_night_thermal)
albedo,albedo_GDAL,_ = dw.read_in_raster(input_albedo)


angular_frequency = 7.27*10**(-5) # rad/sec                maltese et al
day_satellite_overpass_time = 3960000 # 1100*60*60         Scheidt et al
night_satellite_overpass_time = 7956000 # 2210*60*60       Scheidt et al
max_temp_time = 5040000 # 1400*60*60 -> hours to seconds conversion      scheidt et al
min_temp_time = 720000 # 0200*60*60 -> hours to seconds conversion       scheidt et al
latitude = 43.4956
solar_constant = 1367 # W/m**2 for earth        V. M. Fedorov 
Ct_transmittance = 0.75 # atmospheric transmittance for Earth     scheidt et al  
day_temp = raster_day[5,:,:]
night_temp = raster_night[0,:,:]



def calc_max_temp(d_temp, n_temp, max_t_time, d_sat_time, ang_freq, n_sat_time):
    """Calculates maximum temperature based on given variables.

    Args:
        d_temp (arr): daytime temperature
        n_temp (arr): _description_
        max_t_time (int): _description_
        d_sat_time (int): _description_
        ang_freq (float): _description_
        n_sat_time (int): _description_
    """    
    max_temp = (d_temp +((d_temp - n_temp)*((np.cos(ang_freq*max_t_time)) -
    (np.cos(ang_freq*d_sat_time))))/((np.cos(ang_freq*d_sat_time)) -(np.cos(ang_freq*n_sat_time))))
    return max_temp

max_Temperature = calc_max_temp(day_temp, night_temp, max_temp_time, day_satellite_overpass_time, angular_frequency,night_satellite_overpass_time)


min_Temperature = (night_temp +((day_temp - night_temp)*((np.cos(angular_frequency*min_temp_time)) -
(np.cos(angular_frequency*night_satellite_overpass_time))))/((np.cos(angular_frequency*day_satellite_overpass_time))-(np.cos(angular_frequency*night_satellite_overpass_time))))

temp_range = (max_Temperature - min_Temperature)
temp_ratio = (max_Temperature / min_Temperature)

apparent_thermal_inertia = ((1-albedo)/temp_range)

b = ((np.tan(angular_frequency*max_temp_time))/(1-(np.tan(angular_frequency*max_temp_time))))
flight_dt = datetime.fromisoformat('2022-06-17 13:00:00')# Get timezone aware datetime object
flight_dt_tz_aware = dw.get_dt_obj(flight_dt, 'US/Mountain')
solar_declination = np.deg2rad(dw.calc_sol_dec(flight_dt_tz_aware))

phase_diff_1 = (np.arctan(b/(1+b)))     #scheidt et al
phase_diff_2 = (np.arctan((b*np.sqrt(2))/(1+(b*np.sqrt(2)))))   #scheidt et al
xi_constant = (np.arccos(np.tan(solar_declination)*np.tan(np.deg2rad(latitude)))) 

A1_fourier = (((2/np.pi)*(np.sin(solar_declination)*(np.sin(np.deg2rad(latitude))))) + (
    (1/2*np.pi)*(np.cos(solar_declination)*(np.cos(np.deg2rad(latitude))))) * ((np.sin(2*xi_constant) + (2*xi_constant))))

A2_fourier = ((((2*np.sin(solar_declination)*(np.sin(np.deg2rad(latitude))))/(2*np.pi))*(np.sin(2*xi_constant))) + (
    (2*np.cos(solar_declination)*(np.cos(np.deg2rad(latitude)))/(np.pi*(2**2 - 1)))*((2*(np.sin(2*xi_constant))*(np.cos(xi_constant))) - 
    ((np.cos(2*xi_constant))*(np.sin(xi_constant))))))

thermal_inertia = ((apparent_thermal_inertia*((solar_constant*Ct_transmittance)/np.sqrt(angular_frequency)))*(
    ((A1_fourier*((np.cos((angular_frequency*night_satellite_overpass_time)-phase_diff_1))-(np.cos((angular_frequency*day_satellite_overpass_time)-phase_diff_1))))/
    (np.sqrt(1+(1/b)+(1/(2*b**2))))) +
    ((A2_fourier+((np.cos((angular_frequency*night_satellite_overpass_time)-phase_diff_2))-(np.cos((angular_frequency*day_satellite_overpass_time)-phase_diff_2))))/
    (np.sqrt(2+(np.sqrt(2)/b)+(1/(2*b**2)))))))




pos = plt.imshow(thermal_inertia)
plt.colorbar(pos)
plt.show()
print("wow!")



#converting array to tiff file

dw.write_band(albedo_GDAL, thermal_inertia, "E:\\Data\\Georeference_Outputs", "thermal_inertia.tiff", None)
print("done")

dw.write_band(albedo_GDAL, temp_ratio, "E:\\Data\\Georeference_Outputs", "temperature_ratio.tiff", None)
print("done") 