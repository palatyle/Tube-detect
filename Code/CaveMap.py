from tracemalloc import start
import pandas as pd
import geopy

import geopy.distance

fn  = '/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/HHA_tube_bwd.csv'

df = pd.read_csv(fn)


starting_point = geopy.Point(43.49262,-112.44612)


def get_dest_pts(start_pt, dist, bearing):
    """Get destination point given a start point, distance, and bearing.
    
    Args:
        start_pt (geopy.Point): Starting point
        dist (int or float): Distance in meters
        bearing (int or float): Bearing in degrees
        
    Returns:
        new_lat,new_lon (float): Destination points latitude and longitude
    """
    destination = geopy.distance.distance(dist/1000).destination(start_pt, bearing)
    new_lat,new_lon = destination.latitude, destination.longitude
    return new_lat,new_lon

lats = [starting_point[0]]
lons = [starting_point[1]]

for row in df.iloc:
    n_lat, n_lon = get_dest_pts(starting_point, row['Distance'], row['Azimuth'])
    starting_pt = geopy.Point(n_lat,n_lon)
    lats.append(n_lat)
    lons.append(n_lon)
    
df_out= pd.DataFrame({'Latitude':lats, 'Longitude':lons})
# Write lat and lon to csv
# df_out['Lat'] = lats
# df_out['Lon'] = lons
df_out.to_csv('/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/HHA_tube_bwd_geo.csv', index=False)
