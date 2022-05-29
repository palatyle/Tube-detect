import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo
import PIL
from PIL import Image


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
    print("Convert to numpy array...")
    raster_NP = raster.ReadAsArray()
    print("Done!")
    return raster_NP    

def write_band(raster_GDAL, band, dest_dir, out_fn):
    '''
    Returns None. Writes raster band to disk
    Parameters
    ----------
    raster: rasterio object 
    band: Band array to write
    dest_dir: Destination directory 
    out_fn: Output filename
    '''

    if band.dtype == 'int16':
        print('Writing tif...')
        # band = band.filled(fill_value=10001)
        
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_Int16, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.GetRasterBand(1).SetNoDataValue(-32767)
        # dsOut.GetRasterBand(1).SetNoDataValue(np.nan)
        dsOut.FlushCache()
    elif band.dtype == 'uint16':
       
        band = band.filled(fill_value=65535)
        print('Writing tif...')
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_UInt16, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.GetRasterBand(1).SetNoDataValue(65535)
        dsOut.FlushCache()

    elif band.dtype == "float32":
        print('Writing tif...')
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.FlushCache()


    dsOut=None


    return None 


'''
first_file = GDAL_read_tiff("E:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190511T181921_N0212_R127_T12TUP_20190511T224452_super_resolved.tif")
first_raster = GDAL2NP(first_file)

second_file = GDAL_read_tiff("E:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190720T181931_N0213_R127_T12TUP_20190721T001757_super_resolved.tif")
second_raster = GDAL2NP(second_file)

print("done")

test = np.stack([first_raster, second_raster])

test_average = np.average(test, axis=0)

test_std = np.std(test, axis = 0)
print("nice")
'''




working_directory = "E:\Data\HHA_Calculated_Albedo"
os.chdir(working_directory)
working_list = os.listdir(working_directory)
array_list = []

for eachfile in working_list:
    each_gdal = GDAL_read_tiff(eachfile)
    each_raster = GDAL2NP(each_gdal)

    # listtoarray_raster = np.array(each_raster)
    array_list.append(each_raster)

    print("rasters initalized")

    print("done!")


full_stack = np.stack(array_list, axis = 0) 
average = np.average(full_stack, axis = 0)
std_dev = np.std(full_stack, axis = 0)
print("Done!")

fig, ax = plt.subplots()
img_plot = ax.imshow(std_dev)
fig.colorbar(img_plot)
ax.set_xlabel("x distance")
ax.set_ylabel("y diatance")
ax.set_title("Standard Deviation, Hell's Half Acre")

plt.show()

write_band(each_gdal, average, "E:\Data\Georeference_Outputs", "Average.tiff" )
