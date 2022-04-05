import os
import numpy as np
from osgeo import gdal


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

def scale_factor(raster):
    return raster * 10000


    
def albedo_band_math(raster):
    '''
    Returns output of albedo band math
    
    Band cheat sheet:
        
        B1: Coastal Aerosol = raster[0]
        B2: Blue = raster[1]
        B3: Green = raster[2]
        B4: Red = raster[3]
        B5: Red Edge 1 = raster[4]
        B6: Red Edge 2 = raster[5]
        B7: Red Edge 3 = raster[6]
        B8: NIR = raster[7]
        B8A: Narrow NIR = raster[8]
        B9: Water Vapor = raster[9]
        B11: SWIR1 = raster[10]
        B12: SWIR2 = raster[11]
    Parameter
    ---------
    raster: input gdal raster
    
    Returns
    -------
    out: output of albedo band math
    
    '''
    # TODO move outside of function toward end of code prolly

    weight_B2 = 0.2266
    weight_B3 = 0.1236
    weight_B4 = 0.1573
    weight_B8 = 0.3417
    weight_B11 = 0.1170
    weight_B12 = 0.0338

    print("Calculating NDVI...")
    out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
    out = scale_factor((raster[4,:,:].astype(float)-raster[2,:,:].astype(float))/(raster[4,:,:]+raster[2,:,:]))
    print("Done!")
    return out.astype(np.int16)


    #format for albedo calc -- wrap in the scale_factor

    #raster[1,:,:].astype(float)*(0.2266)+raster[2,:,:]*(0.1236) # +...

HHA_files = "D:\\Downloaded_data\\hells_half_acre\\HHA\\Processed_Products\\Forreal_products\\S2A_MSIL2A_20190511T181921_N0212_R127_T12TUP_20190511T224452_super_resolved.tif"

# Read in raster dataset 
src_GDAL = GDAL_read_tiff(HHA_files)

# Get no data value
#nodata = get_no_data_val(src_GDAL)

# Convert GDAL raster dataset to a numpy array
src_NP = GDAL2NP(src_GDAL)

# Apply the no data value to the entire numpy array
#src = apply_no_data_val(src_NP, nodata)
