import os
import numpy as np
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo, BandWriteArray
import data_wrangle as dw
    
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

    print("Calculating NDVI...")
    out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
    out = dw.scale_factor((raster[4,:,:].astype(float)-raster[2,:,:].astype(float))/(raster[4,:,:]+raster[2,:,:]))
    print("Done!")
    return out.astype(np.int16)

def albedo_calculator(raster):
    '''
    Returns calculated albedo for all bands of raster

     Parameters
    ----------
    band_weights: list of weights for needed bands for albedo calculation
    raster: GDAL raster object
    albedo_value: value of calculated albedo
    '''
    band_weights = [0.2266, 0.1236, 0.1573, 0.3417, 0.1170, 0.0338]

    print("Calculating Albedo...")
    albedo_value = raster[1,:,:].astype(float)*band_weights[0]+raster[2,:,:].astype(float)*band_weights[1]+raster[3,:,:].astype(float)*band_weights[2]+raster[7,:,:].astype(float)*band_weights[3]+raster[10,:,:].astype(float)*band_weights[4]+raster[11,:,:].astype(float)*band_weights[5]
    print("Done!")
    return albedo_value

HHA_dir = "E:\\Downloaded_data\\hells_half_acre\\HHA\\Processed_Products\\Forreal_products"
os.chdir(HHA_dir)
file_list = os.listdir(HHA_dir)

for file in file_list:
    print(file)
    # Read in raster dataset 
    src_GDAL = dw.GDAL_read_tiff(file)

    # Get no data value
    #nodata = dw.get_no_data_val(src_GDAL)

    # Convert GDAL raster dataset to a numpy array
    src_NP = dw.GDAL2NP(src_GDAL)

    # Apply the no data value to the entire numpy array
    #src = dw.apply_no_data_val(src_NP, nodata)

    Albedo_Temp = albedo_calculator(src_NP)
    print("Done!")

    dw.write_band(src_GDAL, Albedo_Temp, "E:\\Data\\HHA_Calculated_Albedo", "Albedo" +  file, None) #define outdirectory
    print("Albedo Calcualted")