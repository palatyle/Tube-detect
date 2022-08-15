from statistics import mean
from pysolar import solar
import pytz
from datetime import datetime
import os
import numpy as np
import argparse
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo, BandWriteArray
import time
# import matplotlib.pyplot as plt

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--NDVI",help="Calculate and output NDVI", action="store_true")
parser.add_argument("--NDWI",help="Calculate and output NDWI", action="store_true")
parser.add_argument("--MSAVI2",help="Calculate and output MSAVI2", action="store_true")
parser.add_argument("--thermal",help="Export thermal band", action="store_true")
parser.add_argument("--slope",help="Calculate and output slope", action="store_true")
parser.add_argument("--aspect",help="Calculate and output aspect", action="store_true")
parser.add_argument("--hillshade",help="Calculate and output hillshade", action="store_true")
parser.add_argument("--roughness",help="Calculate and output roughness", action="store_true")
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
    if type(lat) == str:
        lat = float(lat)
        lon = float(lon)
    azi = solar.get_azimuth(lat,lon,dt)
    alt = solar.get_altitude(lat,lon,dt)
    return alt, azi

def get_utc_offset(dt):
    
    return 12 - (dt.tzinfo._utcoffset.seconds / 60 / 60)

def date2jul(dt):
    """Convert datetime object to julian day

    Args:
        dt (datatime obj): Time to convert 

    Returns:
        int: Date in julian day format
    """    
    ordinal_time = dt.toordinal() + 1721425 # convert from propletic gregorian to Julian days (1721425 is offset between these 2 systems)
    return ordinal_time

def julday2julcen(jd):
    """Convert Julian to day to Julian century 

    Args:
        jd (int): Julian day

    Returns:
        float: Julian century
    """    
    return ((jd - 2451545)/36525)

def mean_orb_obliq(jc):
    """Calcualte mean orbital obliquity in degrees for a given julian century value

    Args:
        jc (float): julian century Value

    Returns:
        float: mean orbital obliquity in degrees
    """    
    return 23+(26+((21.448-jc*(46.815+jc*(0.00059-jc*0.001813))))/60)/60
    
def obliq_corr(obl,jc):
    """Calculate corrected obliquity for a given obliquity value and julian century value 

    Args:
        obl (float): mean orbital obliquity in degrees
        jc (float): julian century value

    Returns:
        float: Corrected obliquity in degrees
    """    
    return obl+0.00256*np.cos(np.deg2rad(125.04-1934.136*jc))

def sol_geo_mean_lon(jc):
    """Calculate the mean solar geometric longitude in degrees for a given julian century value

    Args:
        jc (float): julian century value

    Returns:
        float: mean solar geometric longitude in degrees
    """    
    return (280.46646+jc*(36000.76983 + jc*0.0003032)) % 360

def sol_geo_mean_anom(jc):
    """Calculate the mean solar geometric anomaly in degrees for a given julian century value

    Args:
        jc (float): julian century value

    Returns:
        gloat: Mean solar geometric anomaly in degrees
    """    
    return 357.52911+jc*(35999.05029 - 0.0001537*jc)

def sol_eq_ctr(jc,sol_m_anom):
    """Calculate the solar equation of center in degrees for a given julian century value and mean solar geometric anomaly

    Args:
        jc (float): julian century value
        sol_m_anom (float): mean solar geometric anomaly in degrees

    Returns:
        float: solar equation of center
    """    
    return np.sin(np.deg2rad(sol_m_anom))*(1.914602-jc*(0.004817+0.000014*jc))+np.sin(np.deg2rad(2*sol_m_anom))*(0.019993-0.000101*jc)+np.sin(np.deg2rad(3*sol_m_anom))*0.000289

def sol_true_lon(sol_m_lon,sol_ctr):
    """Calculate the solar true longitude in degrees for a given mean solar geometric longitude and solar equation of center

    Args:
        sol_m_lon (float): mean solar geometric longitude in degrees
        sol_ctr (float): solar equation of center

    Returns:
        float: solar true longitude in degrees
    """    
    return (sol_m_lon+sol_ctr)

def sol_app_lon(sol_true_longitude,jc):
    """Calculate the solar apparent longitude in degrees for a given solar true longitude and julian century value

    Args:
        sol_true_longitude (float): solar true longitude in degrees
        jc (float): julian century value

    Returns:
        float: solar apparent longitude in degrees
    """    
    return sol_true_longitude-0.00569-0.00478*np.sin(np.deg2rad(125.04-1934.136*jc))

def sol_decl(sol_app_longitude,corr_obliq):
    """Calculate the solar declination in degrees for a given solar apparent longitude and corrected obliquity

    Args:
        sol_app_longitude (_type_): _description_
        corr_obliq (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(corr_obliq))*np.sin(np.deg2rad(sol_app_longitude))))
    
    
def calc_sol_dec(dt):
    """Calculate solar declination for a given datetime 
    Warning: Does not take timezones other than mountain time into account

    Args:
        dt (datetime obj): given datetime

    Returns:
        float: Solar declination in degrees
    """    
    jul_dat = date2jul(dt)
    jul_dat_cen = julday2julcen(jul_dat)
    orb_obliq = mean_orb_obliq(jul_dat_cen)
    corr_obliquity = obliq_corr(orb_obliq,jul_dat_cen)
    mean_lon_sol = sol_geo_mean_lon(jul_dat_cen)
    mean_sol_anom = sol_geo_mean_anom(jul_dat_cen)
    sol_cent = sol_eq_ctr(jul_dat_cen,mean_sol_anom)
    true_lon_sol = sol_true_lon(mean_lon_sol,sol_cent)
    app_sol_lon = sol_app_lon(true_lon_sol,jul_dat_cen)
    return sol_decl(app_sol_lon,corr_obliquity)

def scale_factor(raster):
    """Apply scale factor of 10000 to raster

    Args:
        raster (numpy arr): raster array

    Returns:
        numpy arr: raster array scaled by 10000
    """    
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
        a = (raster[4,:,:].astype(float)-raster[2,:,:].astype(float)).filled(fill_value=np.nan)
        b = (raster[4,:,:].astype(float)+raster[2,:,:].astype(float)).filled(fill_value=np.nan)
        out = scale_factor(np.divide(a,b,where=b!=0))
        # out = scale_factor((raster[4,:,:].astype(float)-raster[2,:,:].astype(float))/(raster[4,:,:].astype(float)+raster[2,:,:].astype(float)))
        print("Done!")
    elif index == 'NDWI' or index == 'ndwi':
        print("Calculating NDWI...")
        out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
        a = (raster[1,:,:].astype(float)-raster[4,:,:].astype(float)).filled(fill_value=np.nan)
        b = (raster[1,:,:].astype(float)+raster[4,:,:].astype(float)).filled(fill_value=np.nan)
        out = scale_factor(np.divide(a,b,where=b!=0))
        # out = scale_factor((raster[1,:,:].astype(float)-raster[4,:,:].astype(float))/(raster[1,:,:].astype(float)+raster[4,:,:].astype(float)))
        print("Done!")
    elif index == 'MSAVI2' or index == 'msavi2':
        print("Calculating MSAVI2...")
        out = np.zeros(raster[0,:,:].shape, dtype=np.float16)
        out = scale_factor((((2*raster[4,:,:].astype(float))+1) - np.sqrt((((2*raster[4,:,:].astype(float))+1)**2) - (8*(raster[4,:,:].astype(float) - raster[2,:,:].astype(float)))))/2)
        # out = scale_factor((2 * (raster[4,:,:].astype(float) + 1) - np.sqrt((2 * raster[4,:,:].astype(float) + 1)**2 - 8 * (raster[4,:,:].astype(float) - raster[2,:,:].astype(float)))) / 2)
        # out = scale_factor((1/2)*(2*(raster[4,:,:].astype(float)+1)-np.sqrt((2*raster[4,:,:].astype(float)+1)**2-8*(raster[4,:,:].astype(float)-raster[2,:,:].astype(float)))))
        out[out < -1] = -1
        out = out.filled(fill_value=np.nan)
        print("Done!")
    return np.nan_to_num(out,nan=-32767.0).astype(np.int16)
#out.astype(np.int16)
#.filled(fill_value=32767.0).astype(np.int16)


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


    
    if band.dtype == 'int16':
        if arg.CSV:
            numpy2CSV(band,dest_dir,out_fn,-32767)
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
        if arg.CSV:
            numpy2CSV(band,dest_dir,out_fn,65535)
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
        
    elif band.dtype == "float64":
        print('Writing tif...')
        driver = gdal.GetDriverByName("GTiff")
        
        dsOut = driver.Create(os.path.join(dest_dir, out_fn), raster_GDAL.RasterXSize, raster_GDAL.RasterYSize, 1, gdal.GDT_Float64, options=["COMPRESS=LZW"])
        CopyDatasetInfo(raster_GDAL,dsOut)
        dsOut.GetRasterBand(1).WriteArray(band)
        dsOut.GetRasterBand(1).SetNoDataValue(3.4e+38)
        dsOut.FlushCache()
    
    dsOut=None


    return None 

def numpy2CSV(arr, dir, fn ,nodata):

    arr_reshape = arr.reshape(arr.size)
    # arr_no_mask = arr_reshape[arr_reshape != nodata]
    arr_no_mask = arr_reshape
    # arr_no_mask = arr_reshape.compressed()
    print("writing csv...")
    np.savetxt(os.path.join(dir, fn)+'.csv',arr_no_mask,fmt='%i',delimiter=',')
    print("Done")

    return None

def numpy2CSV_float(arr, dir, fn ,nodata):

    arr_reshape = arr.reshape(arr.size)
    # arr_no_mask = arr_reshape[arr_reshape != nodata]
    arr_no_mask = arr_reshape
    # arr_no_mask = arr_reshape.compressed()
    print("writing csv...")
    np.savetxt(os.path.join(dir, fn)+'.csv',arr_no_mask,fmt='%f',delimiter=',')
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
    opts = gdal.DEMProcessingOptions(slopeFormat="degree",computeEdges=True)
    fn = 'DEM_slope.tif'
    slope_gdal = gdal.DEMProcessing(os.path.join(dest_dir, fn),fn_DEM,"slope",options=opts)
    gdal.DEMProcessing(os.path.join(dest_dir, fn),fn_DEM,"slope",options=opts)
    if arg.CSV:
        slope_np = slope_gdal.ReadAsArray()
        numpy2CSV_float(slope_np,dest_dir,fn,-9999.)
    print("Done!")

def calc_roughness(fn_DEM, dest_dir, arg):
    '''
    Returns None. Calculates roughness and writes to disk

    Parameters
    ----------
    fn_DEM: DEM filename
    src_dir: Source directory 
    dest_dir: Destination directory 
    '''

    print('Calculating roughness...')
    opts = gdal.DEMProcessingOptions(computeEdges=True)

    fn = 'DEM_roughness.tif'
    roughness_gdal = gdal.DEMProcessing(os.path.join(dest_dir, fn),fn_DEM,"roughness",options=opts)
    gdal.DEMProcessing(os.path.join(dest_dir, fn),fn_DEM,"roughness")
    
    if arg.CSV:
        roughness_np = roughness_gdal.ReadAsArray()
        numpy2CSV_float(roughness_np, dest_dir, fn, -9999.)
        # tif2csv(fn_DEM, fn, dest_dir)
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
    opts = gdal.DEMProcessingOptions(computeEdges=True)
    fn = 'DEM_aspect.tif'
    aspect_gdal = gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"aspect",options=opts)
    gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"aspect",options=opts)
    if arg.CSV:
        aspect_np = aspect_gdal.ReadAsArray()
        numpy2CSV_float(aspect_np.astype(np.int16), dest_dir, fn, -9999.0)

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
    opts = gdal.DEMProcessingOptions(azimuth=azi, altitude=alt, computeEdges=True)
    fn = 'DEM_hs.tif'
    hs_gdal = gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"hillshade",options=opts)
    gdal.DEMProcessing(os.path.join(dest_dir,fn),fn_DEM,"hillshade",options=opts)
    if arg.CSV:
        hs_np = hs_gdal.ReadAsArray()
        numpy2CSV(hs_np.astype(np.int16), dest_dir, fn, 0)
    print("Done!")

def tif2csv(fn,fn_out,dest):
        np_ras, gdal_ras, nodat = read_in_raster(fn)
        print('Convert to csv...')
        numpy2CSV(np_ras.filled(nodat),dest,fn_out,float(nodat))

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
if args.roughness:
    if args.DEM_dir == None:
        print("Error: Input DEM to calculate slope, aspect, or hillshade")
        quit()
    # Calculate and write roughness data from DEM
    calc_roughness(DEM, d_dir, args)
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

 
    # sol_dec = calc_sol_dec(flight_dt_tz_aware)
    
    # Calculate and write hillshade from DEM. Solar azimuth and altitude calculated from position of tube and time of flight. 
    calc_hillshade(DEM, d_dir, s_azi, s_alt, args)

print("--- %s seconds ---" % (time.time()-start_time))