# connect to the API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os

    


def Sentinel_download(filename, directory):
    
    
    
    api = SentinelAPI('***REMOVED***', '***REMOVED***', 'https://apihub.copernicus.eu/apihub')
    
    # download single scene by known product id
    #api.download(<product_id>)
    
    # search by polygon, time, and SciHub query keywords
    
    #change directory usign os package
    
    os.chdir(directory)
    
    footprint = geojson_to_wkt(read_geojson(filename))
    #loop through date range look around for date range
    
    
    
    years = ["2019", "2020", "2021"]
    for x in years:
        start_date = x + "05" + "01"
        end_date = x + "07" + "31"
        products = api.query(footprint,
                             date=(start_date, end_date),
                             platformname='Sentinel-2',
                             filename = "*MSIL2A*",
                             cloudcoverpercentage=(0, 10))
        
        # download all results from the search
    
        api.download_all(products)
    return None 

direct = "C:\\Users\\***REMOVED***\\Documents\\ArcGIS\\Projects\\Tubez\\HHA"
fn = "C:/Users/***REMOVED***/Documents/ArcGIS/Projects/Tubez/Hell.geojson.json"


Sentinel_download(fn,  direct)
















# GeoJSON FeatureCollection containing footprints and metadata of the scenes
#api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
#api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
#api.get_product_odata(<product_id>)

# Get the product's full metadata available on the server
# api.get_product_odata(<product_id>, full=True) 