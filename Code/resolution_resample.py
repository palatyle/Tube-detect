from os import read
import numpy as np
from pysolar import solar
from datetime import datetime
import pytz
from osgeo import gdal
import operator
import matplotlib.pyplot as plt

import data_wrangle as dw

#dw.get_dt_obj(dt)

'''
#put this into own function -- Still need to do but awaiting instruction

    # Define time zone and datetime of flight
    tzone = 'US/Mountain'

    # Create datetime object
    flight_dt = datetime.fromisoformat(args.date)

    # Get timezone aware datetime object
    flight_dt_tz_aware = dw.get_dt_obj(flight_dt, tzone)
'''

# retrieve day raster resolution
src_day = gdal.Open("E:\\Downloaded_data\\needed_files\\day_ortho_16bit_resample_clip.tif")
xres_day, yres_day = operator.itemgetter(1,5)(src_day.GetGeoTransform())
# xres_day = 0.24024252350136913
# yres_day = -0.24025361345508592
print("done")


# retrieve night raster resoltion ---- this is the one we used earlier 
src_night = gdal.Open("E:\\Downloaded_data\\needed_files\\night_ortho.tif")
xres_night, yres_night = operator.itemgetter(1,5)(src_night.GetGeoTransform())
# xres_night = 0.83713
# yres_night = -0.83713
# _: 65535
# Night seems to  be the coarser raster, so I'll be using that one to resample the albedo
print("done2")


# The night file used earlier and the one below are different. Tyler said to use the following
# in Arc, so here is a sampling of this resolution.
src_night2= gdal.Open("E:\\Downloaded_data\\needed_files\\HHA_night_thermal_only_transparent_mosaic_lwir.tif")
xres_night2, yres_night2 = operator.itemgetter(1,5)(src_night2.GetGeoTransform())
print("done2.1")
# xres_night2 = 0.90539
# yres_night2 = -0.90539
# Seems as if this is the new resolution we should resample to.

#retrieve near.tiff (which is the avg.) resolution. 
src_near = gdal.Open("E:\\Data\\Georeference_Outputs\\near.tiff")
xres_near, yres_near = operator.itemgetter(1,5)(src_near.GetGeoTransform())
print("done3")
# xres_near = 0.83713
# yres_near = -0.83713


# retrieve day DEM resolution
src_dayDEM = gdal.Open("E:\\Downloaded_data\\needed_files\\HHA_day_DEM.tif")
xres_dayDEM, yres_dayDEM = operator.itemgetter(1,5)(src_dayDEM.GetGeoTransform())
print("done4")
# xres_dayDEM = 0.11718799999999964
# yres_dayDEM = -0.11718799999999338





'''
# resample day DEM to be night2's resolution
dayDEM_raster_infn = "E:\\Downloaded_data\\needed_files\\HHA_day_DEM.tif"
dayDEM_raster_outfn = "E:\\Data\\Georeference_Outputs\\HHA_day_DEM_Resample.tif"

xres= xres_night2
yres= yres_night2
resample_alg = 'near'
dayDEM_ds = gdal.Open(dayDEM_raster_infn)
ds = gdal.Warp(dayDEM_raster_outfn, dayDEM_ds, xRes=xres, yRes=yres, resampleAlg=resample_alg)
ds = None
print("DONE")

src_dayDEM_rs = gdal.Open("E:\\Data\\Georeference_Outputs\\HHA_day_DEM_Resample.tif")
xres_dayDEM_rs, yres_dayDEM_rs = operator.itemgetter(1,5)(src_dayDEM_rs.GetGeoTransform())
# xres_dayDEM_rs = 0.90539
# yres_dayDEM_rs = -0.90539
print("done5")
# Successfully resampled
'''




'''
# resample near to be night2's resolution
near_raster_infn = "E:\\Data\\Georeference_Outputs\\near.tiff"
near_raster_outfn = "E:\\Data\\Georeference_Outputs\\near_Resample.tiff"

xres= xres_night2
yres= yres_night2
resample_alg = 'near'
near_ds = gdal.Open(near_raster_infn)
ds = gdal.Warp(near_raster_outfn, near_ds, xRes=xres, yRes=yres, resampleAlg=resample_alg)
ds = None
print("DONE")

src_near_rs = gdal.Open("E:\\Data\\Georeference_Outputs\\near_Resample.tiff")
xres_near_rs, yres_near_rs = operator.itemgetter(1,5)(src_near_rs.GetGeoTransform())
# xres_near_rs = 0.90539
# yres_near_rs = -0.90539
print("done5")
# Successfully resampled
# '''





# resample day to be night2's resolution
day_raster_infn = "E:\\Downloaded_data\\needed_files\\HHA_day_ortho_6band.tif"
day_raster_outfn = "E:\\Data\\Georeference_Outputs\\HHA_day_ortho_6band_resample.tif"

xres= xres_night2
yres= yres_night2
resample_alg = 'near'
day_ds = gdal.Open(day_raster_infn)
ds = gdal.Warp(day_raster_outfn, day_ds, xRes=xres, yRes=yres, resampleAlg=resample_alg)
ds = None
print("DONE")



src_day_rs = gdal.Open("E:\\Data\\Georeference_Outputs\\HHA_day_ortho_6band_resample.tif")
xres_day_rs, yres_day_rs = operator.itemgetter(1,5)(src_day_rs.GetGeoTransform())
print("done5")
# xres_day_rs = 0.90539
# yres_day_rs = -0.90539
# Successfully resampled















'''
# resample albedo to be night's resolution --- OLD CODE
albedo_raster_infn = "E:\\Data\\Georeference_Outputs\\Average.tiff"
albedo_raster_outfn = "E:\\Data\\Georeference_Outputs\\Average_Resample.tiff"

xres= xres_night
yres= yres_night
resample_alg = 'near'
albedo_ds = gdal.Open(albedo_raster_infn)
ds = gdal.Warp(albedo_raster_outfn, albedo_ds, xRes=xres, yRes=yres, resampleAlg=resample_alg)
ds = None
print("done3")

# check to see if albedo has been properly resampled
src_albedo = gdal.Open("E:\\Data\\Georeference_Outputs\\Average_Resample.tiff")
xres_albedo, yres_albedo = operator.itemgetter(1,5)(src_albedo.GetGeoTransform())
# xres_albedo = 0.8371300000001765
# yres_albedo = -0.8371300000001765
print("done4")
'''