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

# upper_dir = "C:\\Users\\***REMOVED***\\Downloads\\"
upper_dir = "/Users/tylerpaladino/Downloads/"

def read_data(filename,no_data_val):
    """Reads in dataframe from csv file. Replaces no data values with nans. 

    Args:
        filename (str): Full filename of csv file to read in.
        no_data_val (int or float): no data value to replace with nan.

    Returns:
        df: dataframe with no data values replaced with nans.
    """    
    df = pd.read_csv(filename,header=None)
    df.replace(no_data_val,np.nan,inplace=True)
    return df 

# Read in dataframes
NDVI_df = read_data(os.path.join(upper_dir+"NDVI.tif.csv"),-32767)

NDWI_df = read_data(os.path.join(upper_dir, "NDWI.tif.csv"),-32767)

thermal_df = read_data(os.path.join(upper_dir, "thermal.tif.csv"),65535)

slope_df = read_data(os.path.join(upper_dir, "DEM_slope.tif.csv"),-9999.)

roughness_df = read_data(os.path.join(upper_dir, "DEM_roughness.tif.csv"),-9999.)

aspect_df = read_data(os.path.join(upper_dir, "DEM_aspect.tif.csv"),-9999.)

hillshade_df = read_data(os.path.join(upper_dir, "DEM_hs.tif.csv"),0)


# Concatenate all the dataframes
full_df = pd.concat([NDVI_df, NDWI_df, slope_df, roughness_df, aspect_df, hillshade_df, thermal_df],axis=1)
full_df.columns = ["NDVI","NDWI","slope","roughness","aspect","hillshade","thermal"]

# Drop any rows that contain nans. 
full_df.dropna(inplace=True)

# Delete dataframes and run garbage collector to free up memory
del [[NDVI_df, NDWI_df, thermal_df, slope_df, roughness_df, aspect_df, hillshade_df]]
gc.collect()
NDVI_df=pd.DataFrame
NDWI_df=pd.DataFrame
thermal_df=pd.DataFrame
slope_df=pd.DataFrame
roughness_df=pd.DataFrame
aspect_df=pd.DataFrame
hillshade_df=pd.DataFrame

regr = linear_model.LinearRegression()
regr.fit(full_df[["NDVI","NDWI","slope","roughness","aspect","hillshade"]],full_df["thermal"])
regr.score(full_df[["NDVI","NDWI","slope","roughness","aspect","hillshade"]],full_df["thermal"])
print(regr.coef_)
# fig,ax=plt.subplots()
# pd.plotting.scatter_matrix(full_df,ax=ax)
# print('Done with pandas plot')
# plt.savefig("C:\\Users\\***REMOVED***\\Downloads\\test.png")



g = sns.pairplot(full_df,corner=True,plot_kws={'alpha':0.2})
g.map_lower(sns.regplot,scatter=False,line_kws={'color':'red','lw':1,'alpha':0.5})
print('Done with seaborn plot command')
g.savefig(os.path.join(upper_dir,"seaborn_scatter.png"))
print("done plotting scatter")
print("--- %s seconds ---" % (time.time()-start_time))

# g2 = sns.pairplot(full_df,kind="kde",corner=True)
# print('Done with seaborn plot command:kde')
# g2.savefig("C:\\Users\\***REMOVED***\\Downloads\\seaborn_kde.png")
# print("done plotting kde")
# print("--- %s seconds ---" % (time.time()-start_time))