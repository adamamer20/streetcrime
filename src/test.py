import osmnx as ox
import os.path
import geopandas as gpd
import random

current_directory = os.path.dirname(__file__)

roads = ox.graph_from_place('University of Toronto, Toronto, Ontario, Canada')

#Saving to geopackage
ox.io.save_graph_geopackage(roads, current_directory+'/roads.gpkg')

#Saving to ml
ox.io.save_graphml(roads, current_directory+'/roads.graphml')

#Loading from geopackage
roads_nodes = gpd.read_file(current_directory+'/roads.gpkg', layer = 'nodes').set_index('osmid') 
roads_edges = gpd.read_file(current_directory+'/roads.gpkg', layer = 'edges').set_index(['u', 'v', 'key']) 
roads_gpkg = ox.utils_graph.graph_from_gdfs(roads_nodes, roads_edges)

#Loading from graphml
roads_graphml = ox.io.load_graphml(current_directory+'/roads.graphml')

#Computing shortest path
origin = random.choice(list(roads.nodes))
destination = random.choice(list(roads.nodes))
path_gpkg = ox.distance.shortest_path(roads_gpkg,
                                      origin,
                                      destination)
print(path_gpkg) #None 
path_graphml = ox.distance.shortest_path(roads_graphml,
                                         origin,
                                         destination)
print(path_graphml) #[389678210, 24959550, 389678187, 389678188, 774054381, 389678189, 2428750571, 389678190, 389678191, 8604490311, 262732431, 389678027, 389678028, 389678029, 3342358877, 779168879, 389677893, 4295105603, 389678009, 389678008, 2078205535, 390545070, 24959546, 306725181, 771950969]
    
