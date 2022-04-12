import os
import numpy as np
import pandas as pd
import time
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo, BandWriteArray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

start_time = time.time()


NDVI_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\NDVI.tif'
NDVI = read_in_tiff_data(NDVI_fn)

# NDWI_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\NDWI.tif'
# NDWI = read_in_tiff_data(NDWI_fn)

# MSAVI2_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\MSAVI2.tif'
# MSAVI2 = read_in_tiff_data(MSAVI2_fn)

# slope_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_slope.tif'
# slope = read_in_tiff_data(slope_fn)

# aspect_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_aspect.tif'
# aspect = read_in_tiff_data(aspect_fn)

# hs_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_hs.tif'
# hs = read_in_tiff_data(hs_fn)

thermal_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\day_ortho_16bit.tif'
thermal = read_thermal_band(thermal_fn)
# Create regression object
reg = LinearRegression(n_jobs=-1)

reg.fit(NDVI,thermal)
print("done")
print("--- %s seconds ---" % (time.time()-start_time))