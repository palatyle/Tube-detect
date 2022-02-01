from matplotlib.colors import rgb_to_hsv
import rasterio as rio
from rasterio.plot import show, show_hist
from pysolar import solar
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
from osgeo import gdal

s_dir = 'C:\Students\Paladino\Tube-detect\data'
d_dir = s_dir
ras = 'day_ortho_16bit.tif'
DEM = 'day_DEM.tif'

tube_lat = 43.169254
tube_lon = -114.34362
tzone = 'US/Mountain'
year = 2021
month = 10
day = 19
hour = 15
minute = 0
second = 0 
microsecond = 0

flight_dt = datetime(year,month,day,hour,minute,second,microsecond)

def get_dt_obj(dt,tzone = 'US/Mountain'):
    tz = pytz.timezone(tzone)
    return tz.localize(flight_dt,is_dst=True)

flight_dt_tz_aware = get_dt_obj(flight_dt)

def get_solar_azi_alt(dt,lat,lon):
    azi = solar.get_azimuth(lat,lon,dt)
    alt = solar.get_altitude(lat,lon,dt)
    return alt, azi

s_alt,s_azi = get_solar_azi_alt(flight_dt_tz_aware,tube_lat, tube_lon)


def wrangle(src_dir, dest_dir, fn_ras, fn_DEM, azi, alt, ndvi_calc=False, ndwi_calc=False, DEM_calcs=True):

    src = rio.open(os.path.join(src_dir,fn_ras))

    if ndwi_calc == True or ndvi_calc == True:
        print("Read in NIR channel")
        NIR = src.read(5, masked=True)
    if ndvi_calc == True:
        print("Read in red channel")
        red = src.read(3, masked=True)
        print("Done!")

        print("Calculating NDVI...")
        NDVI = np.zeros(red.shape, dtype=rio.float32)
        NDVI = (NIR.astype(float)-red.astype(float))/(NIR+red)
        print("Done!")

        print('Writing NDVI data...')
        with rio.Env():
            profile = src.profile
            profile.update(dtype=rio.float32, count=1)
            with rio.open(os.path.join(dest_dir,'NDVI.tif'), 'w', **profile) as dst:
                dst.write(NDVI.astype(float),1)
    if ndwi_calc == True:
        print("Read in green channel")
        green = src.read(2, masked=True)
        print("Done!")

        print("Calculating NDWI...")
        NDWI = np.zeros(red.shape, dtype=rio.float32)
        NDWI = (green.astype(float)-NIR.astype(float))/(green+NIR)
        print("Done!")

        print('Writing NDWI data...')
        with rio.Env():
            profile = src.profile
            profile.update(dtype=rio.float32, count=1)
            with rio.open(os.path.join(dest_dir,'NDWI.tif'), 'w', **profile) as dst:
                dst.write(NDWI.astype(float),1)

    if DEM_calcs == True:
        print('Calculating slope...')
        slope_opts = gdal.DEMProcessingOptions(slopeFormat="degree")
        gdal.DEMProcessing(os.path.join(src_dir,'DEM_slope.tif'),os.path.join(dest_dir,fn_DEM),"slope", options=slope_opts)
        print("Done!")

        print('Calculating aspect...')
        aspect_opts = gdal.DEMProcessingOptions()
        gdal.DEMProcessing(os.path.join(src_dir,'DEM_aspect2.tif'),os.path.join(dest_dir,fn_DEM),"aspect", options=aspect_opts)
        print("Done!")
        
        print('Calculating hillshade...')
        hs_opts = gdal.DEMProcessingOptions(azimuth=azi, altitude=alt)
        gdal.DEMProcessing(os.path.join(src_dir,'DEM_hs.tif'),os.path.join(dest_dir,fn_DEM),"hillshade", options=hs_opts)
        print("Done!")



wrangle(s_dir, d_dir, ras, DEM, s_azi, s_alt, DEM_calcs=True)



# rgb_stack = np.dstack((red,green,blue))
# plt.imshow(rgb_stack,interpolation='none')
# plt.imshow(src.read(),interpolation='none')
# show(src.read())
# for band in range(1, src.count+1):
#     single_band = src.read(band)

#     # get the output name
#     out_name = os.path.basename("day_ortho")
#     file, ext = os.path.splitext(out_name)
#     name = file + "_" + "B" + str(band) + ".tif"
#     out_img = name

#     print(out_img + " done")

#     # Copy the metadata
#     out_meta = src.meta.copy()

#     out_meta.update({"count": 1})

#     os.chdir('/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code')
#     # save the clipped raster to disk
#     with rio.open(out_img, "w", **out_meta) as dest:
#         dest.write(single_band,1)

# b1 = src.read(1)

# with rio.open('b1.tif', "w", **out_meta):

# show(rst)
# plt.imshow(rst.read(),interpolation='none')
# ep.plot_bands(rst,cols=2)
# plt.show()
