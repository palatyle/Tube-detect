import rasterio as rio
from pysolar import solar
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
from osgeo import gdal

# Input directories
# s_dir = 'C:\Students\Paladino\Tube-detect\data'
s_dir = '/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/Code/data'
d_dir = s_dir
ras = 'day_ortho_16bit.tiff'
DEM = 'day_DEM.tiff'

# Define general lat lon coords of tube area
tube_lat = 43.169254
tube_lon = -114.34362

# Define time zone and datetime of flight
tzone = 'US/Mountain'
year = 2021
month = 10
day = 19
hour = 15
minute = 0
second = 0 
microsecond = 0

# Create datetime object
flight_dt = datetime(year,month,day,hour,minute,second,microsecond)


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

def band_math(index, **kwargs):
    '''
    Returns output of a band math index, such as NDVI or NDWI
    
    Parameter
    ---------
    index: band math index (either NDVI or NDWI for now)
    **kawrgs: key value pairs of an arbitrary number of bands in numpy format. Must be of the format b#=...

    Returns
    -------
    out: output of a band math index
    '''
    for idx,val in enumerate(kwargs):
        if val == 'b1':
            f_blue = list(kwargs.values())[idx]
        elif val == 'b2':
            f_green = list(kwargs.values())[idx]
        elif val == 'b3':
            f_red = list(kwargs.values())[idx]
        elif val == 'b4':
            f_red_edge = list(kwargs.values())[idx]
        elif val == 'b5':
            f_NIR = list(kwargs.values())[idx]
        elif val == 'b6':
            f_TIR = list(kwargs.values())[idx]

    if index == 'NDVI' or index ==  'ndvi':
        print("Calculating NDVI...")
        out = np.zeros(f_red.shape, dtype=rio.float32)
        out = (f_NIR.astype(float)-f_red.astype(float))/(f_NIR+f_red)
        print("Done!")
    elif index == 'NDWI' or index == 'ndwi':
        print("Calculating NDWI...")
        out = np.zeros(f_green.shape, dtype=rio.float32)
        out = (f_green.astype(float)-f_NIR.astype(float))/(f_green+f_NIR)
        print("Done!")
    return out


 

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

def write_band(raster, band, dest_dir, out_fn):
    '''
    Returns None. Writes raster band to disk

    Parameters
    ----------
    raster: rasterio object 
    band: Band array to write
    dest_dir: Destination directory 
    out_fn: Output filename
    '''
    print('Writing data...')
    with rio.Env():
        profile = raster.profile
        profile.update(dtype=rio.float32, count=1)
        with rio.open(os.path.join(dest_dir, out_fn), 'w', **profile) as dst:
            dst.write(band.astype(float),1)
    return None 

def calc_slope(fn_DEM, src_dir, dest_dir):
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
    gdal.DEMProcessing(os.path.join(src_dir,'DEM_slope.tif'),os.path.join(dest_dir,fn_DEM),"slope", options=opts)
    print("Done!")

def calc_aspect(fn_DEM, src_dir, dest_dir):
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
    gdal.DEMProcessing(os.path.join(src_dir,'DEM_aspect.tif'),os.path.join(dest_dir,fn_DEM),"aspect", options=opts)
    print("Done!")

def calc_hillshade(fn_DEM, src_dir, dest_dir, azi, alt):
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
    gdal.DEMProcessing(os.path.join(src_dir,'DEM_hs.tif'),os.path.join(dest_dir,fn_DEM),"hillshade", options=opts)
    print("Done!")

# Get timezone aware datetime object
flight_dt_tz_aware = get_dt_obj(flight_dt)

# Get solar azimuth and altitude
s_alt,s_azi = get_solar_azi_alt(flight_dt_tz_aware,tube_lat, tube_lon)

# Create rasterio object
src = rio.open(os.path.join(s_dir,ras))

# Read in bands of interest
red = read_bands(src, 'red')
NIR = read_bands(src, 'NIR')

# Use band math to calculate NDVI 
NDVI = band_math('NDVI', b3=red, b5=NIR)

# Write NDVI data to disk
write_band(src,NDVI,d_dir,'NDVI.tif')

# Calcualte and write slope data from DEM
calc_slope(DEM, s_dir, d_dir)

# Calcualte and write aspect data from DEM
calc_aspect(DEM, s_dir, d_dir)

# Calcualte and write hillshade from DEM. Solar azimuth and altitude calculated from position of tube and time of flight. 
calc_hillshade(DEM, s_dir, d_dir, s_azi, s_alt)