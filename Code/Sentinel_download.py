import os

from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson


def Sentinel_download(filename, directory):
    """Download Sentinel-2 data from Copernicus Hub given a geojson file. 

    Parameters
    ----------
    filename : str
        Path to geojson file containing the polygon to download data for.
    directory : str
        Path to directory to download data to.

    Returns
    -------
    None.
    """    
    
    api = SentinelAPI('user', 'password', 'https://apihub.copernicus.eu/apihub')
    
    os.chdir(directory)
    
    footprint = geojson_to_wkt(read_geojson(filename))
    
    years = ["2022"]
    
    for x in years:
        start_date = x + "05" + "01"
        end_date = x + "08" + "15"
        products = api.query(footprint,
                             date=(start_date, end_date),
                             platformname='Sentinel-2',
                             filename = "*MSIL2A*",
                             cloudcoverpercentage=(0, 10))
        
        # download all results from the search
        api.download_all(products)
    return None 

out = "C:\\Users\\palatyle\\Documents\\Tube-detect\\Sentinel_dl"
fn = "C:\\Users\\palatyle\\Documents\\Tube-detect\\Sentinel_poly.geojson"

Sentinel_download(fn,  out)
















# GeoJSON FeatureCollection containing footprints and metadata of the scenes
#api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
#api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
#api.get_product_odata(<product_id>)

# Get the product's full metadata available on the server
# api.get_product_odata(<product_id>, full=True) 