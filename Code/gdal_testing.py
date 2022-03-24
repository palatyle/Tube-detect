import os
import numpy as np
from osgeo import gdal



os.chdir("C:\\Students\\Paladino\\Tube-detect\\data\\")

filename = "day_ortho_16bit.tif"

raster = gdal.Open(filename)

raster_NP = raster.ReadAsArray()


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