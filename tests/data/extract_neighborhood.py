import geopandas as gpd
import osmnx as ox

neighborhoods = gpd.read_file('data/processed/neighborhoods.gpkg').set_index('id')

roads_nodes = gpd.read_file('data/processed/roads.gpkg', layer = 'nodes').set_index('osmid')
roads_edges = gpd.read_file('data/processed/roads.gpkg', layer = 'edges').set_index(['u', 'v', 'key'])
roads = ox.graph_from_gdfs(roads_nodes, roads_edges)

roads = roads.subgraph(roads_nodes[neighborhoods.loc[1].geometry.contains(roads_nodes.geometry)].index.to_list())
ox.io.save_graph_geopackage(roads, filepath = 'tests/data/processed/roads.gpkg', directed = True)

public_transport_nodes = gpd.read_file('data/processed/public_transport.gpkg', layer = 'nodes').set_index('osmid')
public_transport_edges = gpd.read_file('data/processed/public_transport.gpkg', layer = 'edges').set_index(['u', 'v', 'key'])
public_transport = ox.graph_from_gdfs(public_transport_nodes, public_transport_edges)

public_transport = public_transport.subgraph(public_transport_nodes[neighborhoods.loc[1].geometry.contains(public_transport_nodes.geometry)].index.to_list())
ox.io.save_graph_geopackage(roads, filepath = 'tests/data/processed/public_transport.gpkg', directed = True)

buildings = gpd.read_file('tests/data/processed/buildings.shp').rename(columns = {'neighborho': 'neighborhood'}).set_index('id')
buildings = buildings[buildings['neighborhood'] == 1]
buildings.to_file('tests/data/processed/buildings.shp')




