from tkinter.tix import Meter
import pandas as pd
import geopy

import geopy.distance

fn  = '/Users/tylerpaladino/Documents/cave_pts.csv'

df = pd.read_csv(fn)


starting_point = geopy.Point(40.7128, -74.0060)


def get_dest_pts(start_pt, dist, bearing):
    """Get destination point given a start point, distance, and bearing.
    
    Args:
        start_pt (geopy.Point): Starting point
        dist (int or float): Distance in meters
        bearing (int or float): Bearing in degrees
        
    Returns:
        new_lat,new_lon (float): Destination points latitude and longitude
    """
    destination = geopy.distance.distance(Meter=dist).destination(start_pt, bearing)
    new_lat,new_lon = destination.latitude, destination.longitude
    return new_lat,new_lon

lats = lons = []

for row in df.iloc:
    n_lat, n_lon = get_dest_pts(starting_point, row['dist'], row['bearing'])
    start_pt = geopy.Point(n_lat,n_lon)
    lats.append(n_lat)
    lons.append(n_lon)