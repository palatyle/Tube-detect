import os
import numpy as np
import pandas as pd
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
import gc
import seaborn as sns
# from sklearn.preprocessing import StandardScaler

start_time = time.time()

NDVI_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\NDVI.tif.csv", header=None)
NDVI_df = NDVI_df.replace(-32767,np.nan)
print(len(NDVI_df))

NDWI_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\NDWI.tif.csv",header=None)
NDWI_df = NDWI_df.replace(-32767,np.nan)
print(len(NDWI_df))

thermal_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\thermal.tif.csv",header=None)
thermal_df = thermal_df.replace(65535,np.nan)
print(len(thermal_df))

slope_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\DEM_slope.tif.csv",header=None)
slope_df = slope_df.replace(-9999.,np.nan)
print(len(slope_df))

roughness_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\DEM_roughness.tif.csv",header=None)
roughness_df = roughness_df.replace(-9999.,np.nan)
print(len(roughness_df))

aspect_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\DEM_aspect.tif.csv",header=None)
aspect_df = aspect_df.replace(-9999.,np.nan)
print(len(aspect_df))

hillshade_df = pd.read_csv("C:\\Users\\***REMOVED***\\Downloads\\DEM_hs.tif.csv",header=None)
hillshade_df = hillshade_df.replace(0,np.nan)
print(len(hillshade_df))


full_df = pd.concat([NDVI_df, NDWI_df, slope_df, roughness_df, aspect_df, hillshade_df, thermal_df],axis=1)
full_df.columns = ["NDVI","NDWI","slope","roughness","aspect","hillshade","thermal"]

full_df.dropna(inplace=True)
del [[NDVI_df, NDWI_df, thermal_df, slope_df, roughness_df, aspect_df, hillshade_df]]
gc.collect()
NDVI_df=pd.DataFrame
NDWI_df=pd.DataFrame
thermal_df=pd.DataFrame
slope_df=pd.DataFrame
roughness_df=pd.DataFrame
aspect_df=pd.DataFrame
hillshade_df=pd.DataFrame


# fig,ax=plt.subplots()
# pd.plotting.scatter_matrix(full_df,ax=ax)
# print('Done with pandas plot')
# plt.savefig("C:\\Users\\***REMOVED***\\Downloads\\test.png")



g = sns.pairplot(full_df,corner=True,plot_kws={'alpha':0.2})
print('Done with seaborn plot command')
g.savefig("C:\\Users\\***REMOVED***\\Downloads\\seaborn_scatter.png")
print("done plotting scatter")
print("--- %s seconds ---" % (time.time()-start_time))

g2 = sns.pairplot(full_df,kind="kde",corner=True)
print('Done with seaborn plot command:kde')
g2.savefig("C:\\Users\\***REMOVED***\\Downloads\\seaborn_kde.png")
print("done plotting kde")
print("--- %s seconds ---" % (time.time()-start_time))