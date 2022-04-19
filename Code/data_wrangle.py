from pysolar import solar
import pytz
from datetime import datetime
import os
import numpy as np
import argparse
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo, BandWriteArray
import lidario as lio
import pandas as pd
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--NDVI",help="Calculate and output NDVI", action="store_true")
parser.add_argument("--NDWI",help="Calculate and output NDWI", action="store_true")
parser.add_argument("--MSAVI2",help="Calculate and output MSAVI2", action="store_true")
parser.add_argument("--thermal",help="Export thermal band", action="store_true")
parser.add_argument("--slope",help="Calculate and output slope", action="store_true")
parser.add_argument("--aspect",help="Calculate and output aspect", action="store_true")
parser.add_argument("--hillshade",help="Calculate and output hillshade", action="store_true")
parser.add_argument("--CSV",help="Convert rasters to .csv files", action="store_true")
parser.add_argument("--ortho_dir",help="Full path to multiband ortho raster")
parser.add_argument("--DEM_dir",help="Full path to DEM raster")
parser.add_argument("--out_dir",help="Full path to output directory", default = '/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code/data')
parser.add_argument("--lat",help="Latitude in decimal degrees for hillshade calc", default = 43.169254)
parser.add_argument("--lon",help="Longitude in decimal degrees for hillshade calc", default = -114.34362)
parser.add_argument("--date",help="Date-time in following format = 'yyyy-mm-dd hh:MM:ss' ex: 2021-10-19 15:00:00. Timezone is Mountain time by default", default = '2021-10-19 15:00:00')


args = parser.parse_args()

def get_dt_obj(dt,tzone = 'US/Mountain'):
    '''
    Returns a timezone aware datetime object
    
    Parameters
    ----------
    dt: datetime object
    tzone: time zone string

    Returns
    -------
    Output localized timezone from pytz package
    '''
    tz = pytz.timezone(tzone)
    return tz.localize(dt,is_dst=True)

def get_solar_azi_alt(dt,lat,lon):
    '''
    Returns solar altitude and azimuth using solarpy packages

    Parameters
    ----------
    dt: Timezone aware datetime object
    lat: latitude of general tube location [deg]
    lon: longitude of general tube location [deg]

    Returns
    -------
    alt: altitude of sun from tangential plane of Earth's surface at lat,lon coords 0-90 [deg]
    azi: azimuth of sun from north. 0 to 360 [deg]
    '''
    azi = solar.get_azimuth(lat,lon,dt)
    alt = solar.get_altitude(lat,lon,dt)
    return alt, azi

def scale_factor(raster):
    return raster * 10000

def band_math(raster, index):
    '''
    Returns output of a band math index, such as NDVI or NDWI
    
    Blue = raster[0]
    Green = raster[1]
    Red = raster[2]
    Red Edge = raster[3]
    NIR = raster[4]
    TIR = raster[5]

    Parameter
    ---------
    index: band math index (either NDVI or NDWI for now)

    Returns
    -------
    out: output of a band math index

    
    '''
    if index == 'NDVI' or index ==  'ndvi':
        print("Calculating NDVI...")
        out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
        out = scale_factor((raster[4,:,:].astype(float)-raster[2,:,:].astype(float))/(raster[4,:,:]+raster[2,:,:]))
        print("Done!")
    elif index == 'NDWI' or index == 'ndwi':
        print("Calculating NDWI...")
        out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
        out = scale_factor((raster[1,:,:].astype(float)-raster[4,:,:].astype(float))/(raster[1,:,:]+raster[4,:,:]))
        print("Done!")
    elif index == 'MSAVI2' or index == 'msavi2':
        print("Calculating MSAVI2...")
        out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
        # out = scale_factor((((2*raster[4,:,:].astype(float))+1) - np.sqrt((((2*raster[4,:,:].astype(float))+1)**2) - (8*(raster[4,:,:].astype(float) - raster[2,:,:].astype(float)))))/2)
        out = scale_factor((1/2)*(2*(raster[4,:,:].astype(float)+1)-np.sqrt((2*raster[4,:,:].astype(float)+1)**2-8*(raster[4,:,:].astype(float)-raster[2,:,:].astype(float)))))
        print("Done!")
    return out.astype(np.int16)


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
    print("reading in GDAL obj...")
    raster = gdal.Open(fn)
    print("Done!")
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

def read_bands(raster, band_name):
    '''
    Returns band as a masked numpy array specified by band name

    Parameters
    ----------
    raster: rasterio object 
    band_name: String containing name of band
    '''
    if band_name == 'blue':
        print("Read in blue channel...")
        out = raster.read(1,masked=True)
        print('Done!')
    elif band_name == 'green':
        print("Read in green channel...")
        out = raster.read(2,masked=True)
        print('Done!')
    elif band_name == 'red':
        print("Read in red channel...")
        out = raster.read(3,masked=True)
        print('Done!')
    elif band_name == 'red edge' or band_name == 'red_edge' or band_name == 'rededge':
        print("Read in red edge channel...")
        out = raster.read(4,masked=True)
        print('Done!')
    elif band_name == 'nir' or band_name == 'NIR':
        print("Read in NIR channel...")
        out = raster.read(5,masked=True)
        print('Done!')
    elif band_name == 'tir' or band_name == 'TIR':
        print("Read in TIR channel...")
        out = raster.read(6,masked=True)
        print('Done!')
    return out

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

    # Return no data value rounded to nearest integer value
    return round(no_data)

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

def write_band(raster_GDAL, band, dest_dir, out_fn, arg):
    '''
    Returns None. Writes raster band to disk

    Parameters
    ----------
    raster: rasterio object 
    band: Band array to write
    dest_dir: Destination directory 
    out_fn: Output filename
    '''

    if arg.CSV:
        numpy2CSV(band,dest_dir,out_fn)
    
    if band.dtype == 'int16':
        print('Writing tif...')
        band = band.filled(fill_value=10001)
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_Int16, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.GetRasterBand(1).SetNoDataValue(10001)
        dsOut.FlushCache()
    elif band.dtype == 'uint16':
        print('Writing tif...')
        band = band.filled(fill_value=65535)
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_UInt16, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.GetRasterBand(1).SetNoDataValue(65535)
        dsOut.FlushCache()

    dsOut=None


    return None 

def numpy2CSV(arr, dir, fn):

    arr_reshape = arr.reshape(arr.size)
    arr_no_mask = arr_reshape.compressed()
    print("writing csv...")
    np.savetxt(os.path.join(dir, fn)+'.csv',arr_no_mask,fmt='%i',delimiter=',')
    print("Done")

    return None

def calc_slope(fn_DEM, dest_dir, arg):
    '''
    Returns None. Calculates slope and writes to disk

    Parameters
    ----------
    fn_DEM: DEM filename
    src_dir: Source directory 
    dest_dir: Destination directory 
    '''

    print('Calculating slope...')
    opts = gdal.DEMProcessingOptions(slopeFormat="degree")
    fn = 'DEM_slope.tif'
    gdal.DEMProcessing(os.path.join(dest_dir, fn),fn_DEM,"slope",options=opts)
    if arg.CSV:
        tif2csv(fn_DEM, fn, dest_dir)
    print("Done!")

def calc_aspect(fn_DEM, dest_dir, arg):
    '''
    Returns None. Calculates aspect and writes to disk

    Parameters
    ----------
    fn_DEM: DEM filename
    src_dir: Source directory 
    dest_dir: Destination directory 
    '''
    print('Calculating aspect...')
    opts = gdal.DEMProcessingOptions()
    fn = 'DEM_aspect.tif'
    gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"aspect",options=opts)
    if arg.CSV:
        tif2csv(fn_DEM, fn, dest_dir)
    print("Done!")

def calc_hillshade(fn_DEM, dest_dir, azi, alt, arg):
    '''
    Returns None. Calculates a hillshade and writes to disk

    Parameters
    ----------
    fn_DEM: DEM filename
    src_dir: Source directory 
    dest_dir: Destination directory 
    azi: solar azimuth 0-360 [deg]
    alt: Solar altitude 0-90 [deg]
    '''
    print('Calculating hillshade...')
    opts = gdal.DEMProcessingOptions(azimuth=azi, altitude=alt)
    fn = 'DEM_hs.tif'
    gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"hillshade",options=opts)
    if arg.CSV:
        tif2csv(fn_DEM, fn, dest_dir)
    print("Done!")

def tif2csv(fn,fn_out,dest):
        np_ras, gdal_ras, nodat = read_in_raster(fn)
        print('Convert to csv...')
        numpy2CSV(np_ras,dest,fn_out)

def read_in_raster(fn):
    '''
    
    
    '''    
    # Read in raster dataset 
    src_GDAL = GDAL_read_tiff(fn)

    # Get no data value
    nodata = get_no_data_val(src_GDAL)

    # Convert GDAL raster dataset to a numpy array
    src_NP = GDAL2NP(src_GDAL)

    # Apply the no data value to the entire numpy arr
    src = apply_no_data_val(src_NP, nodata)

    # Free up some memory
    src_NP = None

    return src, src_GDAL, nodata


# Input directories
# s_dir = 'C:\Students\Paladino\Tube-detect\data'
ras = args.ortho_dir
DEM = args.DEM_dir
d_dir = args.out_dir

# ras = 'day_ortho_16bit.tiff'
# DEM = 'day_DEM.tiff'

if args.NDVI or args.NDWI or args.MSAVI2 or args.thermal:

    raster,raster_GDAL,nodat = read_in_raster(ras)

    # If any of these above indices are in the arg list, load in NIR (every index depends on NIR)
    if args.NDVI:
        # Use band math to calculate NDVI 
        NDVI = band_math(raster, 'NDVI')
        # Write NDVI data to disk
        write_band(raster_GDAL,NDVI,d_dir,'NDVI.tif',args)
        NDVI = None
    if args.MSAVI2:
        # Use band math to calculate MSAVI2
        MSAVI2 = band_math(raster, 'MSAVI2')
        # Write MSAVI2 data to disk
        write_band(raster_GDAL,MSAVI2,d_dir,'MSAVI2.tif',args)
        MSAVI2 = None
    if args.NDWI:
        # Use band math to calculate MSAVI2
        NDWI = band_math(raster, 'NDWI')
        # Write MSAVI2 data to disk
        write_band(raster_GDAL,NDWI,d_dir,'NDWI.tif',args)
        NDWI = None
    if args.thermal:
        write_band(raster_GDAL,raster[-1,:,:],d_dir,'thermal.tif',args)

if args.slope:
    if args.DEM_dir == None:
        print("Error: Input DEM to calculate slope, aspect, or hillshade")
        quit()
    # Calculate and write slope data from DEM
    calc_slope(DEM, d_dir, args)

if args.aspect:
    if args.DEM_dir == None:
        print("Error: Input DEM to calculate slope, aspect, or hillshade")
        quit()
    # Calculate and write aspect data from DEM
    calc_aspect(DEM, d_dir, args)

if args.hillshade:
    if args.DEM_dir == None:
        print("Error: Input DEM to calculate slope, aspect, or hillshade")
        quit()
    # Define general lat lon coords of tube area
    tube_lat = args.lat
    tube_lon = args.lon

    # Define time zone and datetime of flight
    tzone = 'US/Mountain'

    # Create datetime object
    flight_dt = datetime.fromisoformat(args.date)

    # Get timezone aware datetime object
    flight_dt_tz_aware = get_dt_obj(flight_dt, tzone)

    # Get solar azimuth and altitude
    s_alt,s_azi = get_solar_azi_alt(flight_dt_tz_aware,tube_lat,tube_lon)

    # Calcualte and write hillshade from DEM. Solar azimuth and altitude calculated from position of tube and time of flight. 
    calc_hillshade(DEM, d_dir, s_azi, s_alt, args)

print("--- %s seconds ---" % (time.time()-start_time))