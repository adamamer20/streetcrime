import osmnx as ox
import networkx as nx
import geopandas as gpd
import os

directory = os.path.dirname(__file__)
for _ in range(1):
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory
    
roads_nodes = gpd.read_file(os.path.join(directory, r"data\processed\roads.gpkg"), layer = 'nodes').set_index('osmid')
roads_edges = gpd.read_file(os.path.join(directory, r"data\processed\roads.gpkg"), layer = 'edges').set_index(['u', 'v', 'key'])
roads = ox.graph_from_gdfs(roads_nodes, roads_edges)

public_transport_nodes = gpd.read_file(os.path.join(directory, r"data\processed\public_transport.gpkg"), layer='nodes').set_index('osmid')
public_transport_edges = gpd.read_file(os.path.join(directory, r"data\processed\public_transport.gpkg"), layer='edges').set_index(['u', 'v', 'key'])
public_transport = ox.graph_from_gdfs(public_transport_nodes, public_transport_edges)


roads_cc = max(nx.weakly_connected_components(roads), key=len)
public_transport_cc = max(nx.weakly_connected_components(public_transport), key=len)

roads = roads.subgraph(roads_cc)
public_transport = public_transport.subgraph(public_transport_cc)

ox.save_graph_geopackage(roads, filepath=os.path.join(directory, r"data\processed\roads_consolidated.gpkg"), directed=True)
ox.save_graph_geopackage(public_transport, filepath=os.path.join(directory, r"data\processed\public_transport_consolidated.gpkg"), directed=True)