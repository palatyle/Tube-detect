# connect to the API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date 
import os


'''
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date))):
        yield start_date + timedelta(n)
        
start_date = date(2021, 5, 1)
end_date = date(2021, 5, 30)
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d")) 
    '''



api = SentinelAPI('***REMOVED***', '***REMOVED***', 'https://apihub.copernicus.eu/apihub')

# download single scene by known product id
#api.download(<product_id>)

# search by polygon, time, and SciHub query keywords
footprint = geojson_to_wkt(read_geojson('C:/Users/***REMOVED***/Documents/ArcGIS/Projects/Tubez/Hell.geojson.json')) #file pathway to geojson files
#loop through date range look around for date range
products = api.query(footprint,
                     date=('20210501', '20210530'),
                     platformname='Sentinel-2',
                     filename = "*MSIL2A*",
                     cloudcoverpercentage=(0, 10))

# download all results from the search
#change directory usign os package

os.chdir('C:\\Users\\***REMOVED***\\Documents\\ArcGIS\\Projects\\Tubez\\HHA')


api.download_all(products)
















# GeoJSON FeatureCollection containing footprints and metadata of the scenes
#api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
#api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
#api.get_product_odata(<product_id>)

# Get the product's full metadata available on the server
#api.get_product_odata(<product_id>, full=True) 