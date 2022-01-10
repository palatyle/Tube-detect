'''
Detection code outline
'''

def main():
    '''
    
    '''

def dl_img(): #easyish (needs to work in gnu parallel)
    '''
    Downloads image + aux data from webserver
    Returns image matrix & aux data
    '''

def thermal_filter(): #hard
    '''
    Follows work by Karki to remove background thermal effect s. Output will be residual thermal effect not from extraneous processes. 
    Terrestrial and lunar version
    '''
    
def albedo_calc():
    '''
    Download many Sentinel 2A images of study area, supersample lower resolution bands, then calculate albedo. Download images from same time of year. How much does albedo change? 
    
    1. Download imagery (https://github.com/sentinel-hub/sentinelhub-py)
    2. Crop imagery to study area
    3. Supersample coarse bands
    4. Calculate albedo
    
    Albedo equation for Sentinel 2 data: https://ieeexplore.ieee.org/abstract/document/8974188

    Supersampling algorithm (paper): https://ieeexplore.ieee.org/ielx7/36/7987135/07924391.pdf?tp=&arnumber=7924391&isnumber=7987135&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50Lzc5MjQzOTE=
    Supersampling algorithm (code): https://nicolas.brodu.net/recherche/superres/
    '''

def process_img(): # easyish 
    '''
    Inputs: 
    day image
    night image
    auxilary information (lat, lon, solar declination, solar declination) - Likely need to calculate these values for UAS imagery
    Returns thermal inertia image product using Xue and Cracknell eqs 
    '''
def crater_removal(): # easy
    '''
    Use Robbin's database of craters to filter image
    '''

def find_tubes(): # hard
    '''
    Uses edge detections, other convolutions or other computer vision techniques to ID tube locations in thermal inertia image. 
    '''