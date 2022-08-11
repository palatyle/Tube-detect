import os
import numpy as np
import pandas as pd
import time
import data_wrangle as dw
from sklearn import linear_model
# from sklearn.preprocessing import StandardScaler

start_time = time.time()

NDVI_df = pd.read_csv('/Users/tylerpaladino/Downloads/NDVI.tif.csv',header=None)
NDVI_df = NDVI_df.replace(-32767,np.nan)
print(len(NDVI_df))

NDWI_df = pd.read_csv('/Users/tylerpaladino/Downloads/NDWI.tif.csv',header=None)
NDWI_df = NDWI_df.replace(-32767,np.nan)
print(len(NDWI_df))

thermal_df = pd.read_csv('/Users/tylerpaladino/Downloads/thermal.tif.csv',header=None)
thermal_df = thermal_df.replace(65535,np.nan)
print(len(thermal_df))

slope_df = pd.read_csv('/Users/tylerpaladino/Downloads/DEM_slope.tif.csv',header=None)
slope_df = slope_df.replace(-9999.,np.nan)
print(len(slope_df))

roughness_df = pd.read_csv('/Users/tylerpaladino/Downloads/DEM_roughness.tif.csv',header=None)
roughness_df = roughness_df.replace(-9999.,np.nan)
print(len(roughness_df))

aspect_df = pd.read_csv('/Users/tylerpaladino/Downloads/DEM_aspect.tif.csv',header=None)
aspect_df = aspect_df.replace(-9999.,np.nan)
print(len(aspect_df))

hillshade_df = pd.read_csv('/Users/tylerpaladino/Downloads/DEM_hs.tif.csv',header=None)
hillshade_df = hillshade_df.replace(0,np.nan)
print(len(hillshade_df))


full_df = pd.concat([NDVI_df, NDWI_df, thermal_df, slope_df, roughness_df, aspect_df, hillshade_df],axis=1)
full_df.columns = ["NDVI","NDWI","thermal","slope","roughness","aspect","hillshade"]
pd.plotting.scatter_matrix(full_df)
print("done")
print("--- %s seconds ---" % (time.time()-start_time))