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

