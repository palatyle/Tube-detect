from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import rasterio as rio
from pysolar import solar
import pytz
from datetime import datetime
import os
import numpy as np
import argparse
from osgeo import gdal
import lidario as lio

parser = argparse.ArgumentParser()
parser.add_argument("--NDVI",help="Calculate and output NDVI", action="store_true")
parser.add_argument("--NDWI",help="Calculate and output NDWI", action="store_true")
parser.add_argument("--MSAVI2",help="Calculate and output MSAVI2", action="store_true")
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
    elif index == 'MSAVI2' or index == 'msavi2':
        print("Calculating MSAVI2...")
        out = np.zeros(f_red.shape, dtype=rio.float32)
        out = (((2*f_NIR.astype(float))+1) - np.sqrt((((2*f_NIR.astype(float))+1)**2) - (8*(f_NIR.astype(float) - f_red.astype(float)))))/2
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

def write_band(raster, band, dest_dir, out_fn, arg):
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

    if arg.CSV:
        print('Convert to csv...')
        # Create Translator object
        translator = lio.Translator("geotiff", "csv")
        # Read in tiff and output a .csv file. 
        translator.translate(os.path.join(dest_dir, out_fn), out_file=os.path.join(dest_dir, out_fn) + '.csv')
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
        print('Convert to csv...')
        # Create Translator object
        translator = lio.Translator("geotiff", "csv")
        # Read in tiff and output a .csv file. 
        translator.translate(os.path.join(dest_dir,fn), out_file=os.path.join(dest_dir, fn) + '.csv')
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
        print('Convert to csv...')
        # Create Translator object
        translator = lio.Translator("geotiff", "csv")
        # Read in tiff and output a .csv file. 
        translator.translate(os.path.join(dest_dir,fn), out_file=os.path.join(dest_dir, fn) + '.csv')
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
        print('Convert to csv...')
        # Create Translator object
        translator = lio.Translator("geotiff", "csv")
        # Read in tiff and output a .csv file. 
        translator.translate(os.path.join(dest_dir,fn), out_file=os.path.join(dest_dir, fn) + '.csv')
    print("Done!")



# Input directories
# s_dir = 'C:\Students\Paladino\Tube-detect\data'
ras = args.ortho_dir
DEM = args.DEM_dir
d_dir = args.out_dir

# ras = 'day_ortho_16bit.tiff'
# DEM = 'day_DEM.tiff'

if args.NDVI or args.NDWI or args.MSAVI2:
    # Create rasterio object
    src = rio.open(ras)

    # If any of these above indices are in the arg list, load in NIR (every index depends on NIR)
    NIR = read_bands(src, 'NIR')
    if args.NDVI or args.MSAVI2:
        red = read_bands(src, 'red')
        if args.NDVI:
            # Use band math to calculate NDVI 
            NDVI = band_math('NDVI', b3=red, b5=NIR)
            # Write NDVI data to disk
            write_band(src,NDVI,d_dir,'NDVI.tif',args)
        if args.MSAVI2:
            # Use band math to calculate MSAVI2
            MSAVI2 = band_math('MSAVI2', b3=red, b5=NIR)
            # Write MSAVI2 data to disk
            write_band(src,MSAVI2,d_dir,'MSAVI2.tif',args)
    elif args.NDWI:
        green = read_bands(src, 'green')
        # Use band math to calculate MSAVI2
        NDWI = band_math('NDWI', b2=green, b5=NIR)
        # Write MSAVI2 data to disk
        write_band(src,NDWI,d_dir,'NDWI.tif',args)
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


