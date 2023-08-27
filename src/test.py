import scipy
import geopandas as gpd
import os.path
directory = os.path.dirname(__file__)
parent_directory = os.path.split(directory)[0]
directory = parent_directory
    
neighborhoods = gpd.read_file(directory + r"\data\processed\neighborhoods.gpkg")
neighborhoods.set_index('id', inplace=True)
scipy.stats.skewnorm(a = ae, loc = loce, scale = scalee)

