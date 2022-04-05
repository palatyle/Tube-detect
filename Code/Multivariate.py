import os
import numpy as np
import pandas as pd
import time
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo, BandWriteArray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

start_time = time.time()

def GDAL_read_tiff(fn):
    '''
    Returns GDAL raster object

    Parameters
    ----------
    fn: Full directory and filename of .tiff 

    Returns
    -------
    raster: GDAL raster object
    '''
    raster = gdal.Open(fn)
    return raster

def get_no_data_val(ds):
    '''
    Returns no data value present in GDAL raster dataset

    Parameters
    ----------
    ds: GDAL raster object

    Returns
    -------
    no data value in uint16 format
    '''
    # Read in 1st band to get access to nodata value
    dummy_band = ds.GetRasterBand(1)
    no_data = dummy_band.GetNoDataValue()

    # Return no data value as a unsigned 16 bit integer (to match input dataset)
    return np.uint16(no_data)


def GDAL2NP(raster):
    '''
    Returns N dimensional numpy array of GDAL raster object

    Parameters
    ----------
    raster: GDAL raster object

    Returns
    -------
    raster_NP: raster numpy array
    '''
    raster_NP = raster.ReadAsArray()
    return raster_NP

def apply_no_data_val(ds, no_data):
    '''
    Returns numpy array with no data values masked out

    Parameters
    ----------
    ds: Numpy raster object 
    no_data: no data value

    Returns
    -------
    Masked numpy array of original raster data
    '''
    return np.ma.masked_equal(ds,no_data)

def read_thermal_band(fn):
    src_GDAL = GDAL_read_tiff(fn)
    band = src_GDAL.GetRasterBand(6).ReadAsArray()
    nodata = get_no_data_val(src_GDAL)

    # Convert GDAL raster dataset to a numpy array
    
    return apply_no_data_val(band, nodata)

def read_in_tiff_data(fn):
    src_GDAL = GDAL_read_tiff(fn)

    # Get no data value
    nodata = get_no_data_val(src_GDAL)

    # Convert GDAL raster dataset to a numpy array
    src_NP = GDAL2NP(src_GDAL)

    # Apply the no data value to the entire numpy arr
    return apply_no_data_val(src_NP, nodata)

start_time = time.time()

NDVI_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\NDVI.tif'
NDVI = read_in_tiff_data(NDVI_fn)

NDWI_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\NDWI.tif'
NDWI = read_in_tiff_data(NDWI_fn)

MSAVI2_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\MSAVI2.tif'
MSAVI2 = read_in_tiff_data(MSAVI2_fn)

slope_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_slope.tif'
slope = read_in_tiff_data(slope_fn)

aspect_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_aspect.tif'
aspect = read_in_tiff_data(aspect_fn)

hs_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\DEM_hs.tif'
hs = read_in_tiff_data(hs_fn)

thermal_fn = 'C:\\Students\\Paladino\\Tube-detect\\data\\day_ortho_16bit.tif'
thermal = read_thermal_band(thermal_fn)
# Create regression object
reg = LinearRegression()

reg.fit(NDVI,thermal)
print("done")
print("--- %s seconds ---" % (time.time()-start_time))