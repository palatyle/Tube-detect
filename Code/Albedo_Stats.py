import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal
<<<<<<< HEAD
import PIL
from PIL import Image


'''
ims = []
ims.append(Image.open("D:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190511T181921_N0212_R127_T12TUP_20190511T224452_super_resolved.tif"))
'''


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

for eachfile in working_list:
    each_gdal = GDAL_read_tiff(eachfile)
    each_raster = GDAL2NP(each_gdal)

    print("rasters initalized")

    full_stack = np.stack(each_raster, axis = 0)  #i saw the .shape online, dunno if I need it

    print("done!")

average = np.average(full_stack, axis = 0)
std_dev = np.std(full_stack, axis = 0)
=======

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

first_file = GDAL_read_tiff("D:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190511T181921_N0212_R127_T12TUP_20190511T224452_super_resolved.tif")
first_raster = GDAL2NP(first_file)

second_file = GDAL_read_tiff("D:\Data\HHA_Calculated_Albedo\AlbedoS2A_MSIL2A_20190720T181931_N0213_R127_T12TUP_20190721T001757_super_resolved.tif")
second_raster = GDAL2NP(second_file)

print("done")

test = np.stack([first_raster, second_raster])

test_average = np.average(test, axis=0)

test_std = np.std(test, axis = 0)
print("nice")


>>>>>>> dc5d8b9a33c905c5ddc6f9d7b0b1b999843330ea
