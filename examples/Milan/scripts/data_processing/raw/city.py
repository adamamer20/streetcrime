from ast import literal_eval
import geopandas as gpd
import pandas as pd
import os.path
import shapely
from shapely import intersects, intersects_xy
from fiona.errors import DriverError
import time
import os.path
import osmnx as ox
import networkx as nx
import dbfread
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy import array, argmin

directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory
    
def obtaining_roads(city_name: str, crs: str, out_file: str) -> nx.MultiDiGraph:
    # Downloading the full dataset of roads in Milan
    roads = ox.graph_from_place(city_name)  # simplifaction is already activated
    roads = ox.projection.project_graph(roads, to_crs=crs)
    roads = ox.simplification.consolidate_intersections(roads, tolerance=20)
    roads = ox.speed.add_edge_speeds(roads)
    edges = ox.graph_to_gdfs(roads, nodes=False, edges=True)
    edges["speed_kph"] = edges["speed_kph"]*0.6
    edges["attributes_dict"] = "{'speed_kph': " + edges["speed_kph"].astype(str) + "}"
    edges["attributes_dict"] = edges["attributes_dict"].apply(literal_eval)
    attributes_dict = edges["attributes_dict"].to_dict()
    nx.set_edge_attributes(roads, attributes_dict)
    #edges_speed = zip(edges.index, edges['speed_kph']})
    #edges_speed = dict(edges_speed)
    roads = ox.speed.add_edge_travel_times(roads)
    #ox.io.save_graphml(roads, filepath=out_file)
    ox.io.save_graph_geopackage(roads, filepath = out_file, directed = True)
    return roads

def neighborhoods_data_processing(
    raw_file: str, pop_file: str, income_file: str, cap_file: str, out_file: str
) -> gpd.GeoDataFrame:
    # Read files
    neighborhoods = gpd.read_file(raw_file)
    neighborhoods_pop = pd.read_csv(pop_file)
    cap_income = pd.read_csv(income_file)
    cap = gpd.read_file(cap_file)
    # Drop unnecessary columns
    neighborhoods = neighborhoods[["ID_NIL", "NIL", "geometry"]]
    neighborhoods_pop = neighborhoods_pop[["NIL", "Totale", "Totale_Milano"]]
    cap = cap[["POSTCODE", "geometry"]]
    # Fix types
    cap["POSTCODE"] = cap["POSTCODE"].astype(int)
    # Compute proportion of population in each neighborhood
    neighborhoods_pop["prop"] = neighborhoods_pop["Totale"] / neighborhoods_pop[
        "Totale_Milano"
    ].astype(int)
    neighborhoods_pop = neighborhoods_pop[["NIL", "prop"]]
    # Rename columns to match other df
    neighborhoods_pop.rename(columns={"NIL": "ID_NIL"}, inplace=True)
    cap_income.rename(columns={"CAP": "POSTCODE"}, inplace=True)
    # Merge dataframes
    neighborhoods = neighborhoods.merge(neighborhoods_pop, on="ID_NIL", how="left")
    cap = cap.merge(cap_income, on="POSTCODE", how="left")
    # Set indexes
    cap.set_index("POSTCODE", inplace=True)
    neighborhoods.set_index("ID_NIL", inplace=True)
    # Find in which neighborhoods are cap located
    neighborhoods["cap"] = None
    for (
        id,
        neighborhood,
    ) in (
        neighborhoods.iterrows()
    ):  # TODO: do it with list comprehension (look at set building entrance)
        neighborhoods.at[id, "cap"] = find_intersecting_cap(neighborhood.geometry, cap)

    # Add frequency and mean income per neighborhood
    frequency_columns = [
        column for column in cap_income.columns.str.split(" ") if "Frequenza" in column
    ]
    frequency_columns = [" ".join(column) for column in frequency_columns]
    mean_columns = [
        column for column in cap_income.columns.str.split(" ") if "Media" in column
    ]
    mean_columns = [" ".join(column) for column in mean_columns]
    city_data = []
    for id, neighborhood in neighborhoods.iterrows():
        neighborhood_data = []
        neighborhoods.at[id, "sum income"] = 0
        for i in range(len(frequency_columns)):
            frequency_per_cap = []
            mean_per_cap = []
            for postcode in neighborhood["cap"]:
                frequency_per_cap.append(cap[frequency_columns[i]].loc[postcode])
                mean_per_cap.append(cap[mean_columns[i]].loc[postcode])
            neighborhoods.at[id, mean_columns[i]] = np.average(
                mean_per_cap, weights=frequency_per_cap
            )
            neighborhoods.at[id, frequency_columns[i]] = sum(frequency_per_cap)
            neighborhood_data.append(
                [
                    neighborhoods.at[id, mean_columns[i]]
                    for _ in range(int(neighborhoods.at[id, frequency_columns[i]]))
                ]
            )
        neighborhood_data = [mean for bins in neighborhood_data for mean in bins]
        city_data.append(neighborhood_data)
        neighborhood_data_df = pd.DataFrame(neighborhood_data)
        ae, loce, scalee = scipy.stats.skewnorm.fit(neighborhood_data_df)
        neighborhoods.at[id, "ae"] = ae
        neighborhoods.at[id, "loce"] = loce
        neighborhoods.at[id, "scalee"] = scalee
        neighborhoods.at[id, 'sum income'] += neighborhoods.at[id, mean_columns[i]] * neighborhoods.at[id, frequency_columns[i]]
    city_data = [mean for neighborhood in city_data for mean in neighborhood]
    city_data_df = pd.DataFrame(city_data)
    city_ae, city_loce, city_scalee = scipy.stats.skewnorm.fit(city_data_df)
    neighborhoods["city_ae"] = city_ae
    neighborhoods["city_loce"] = city_loce
    neighborhoods["city_scalee"] = city_scalee
    neighborhoods['sum frequency'] = neighborhoods[frequency_columns].sum(axis = 1)
    neighborhoods['mean income'] = neighborhoods['sum income'] / neighborhoods['sum frequency']

    # Drop unnecessary columns
    neighborhoods.drop(['sum income', 'sum frequency'], axis = 1, inplace = True)
    neighborhoods.drop(frequency_columns, axis=1, inplace=True)
    neighborhoods.drop(mean_columns, axis=1, inplace=True)

    # Rename columns
    neighborhoods.rename(columns={"NIL": "name"}, inplace=True)
    neighborhoods.rename_axis("id", inplace=True)

    # Change CRS to epsg:7791
    neighborhoods.to_crs("epsg:7791", inplace=True)
    # Convert CAP list to string (to save the file)
    neighborhoods["cap"] = neighborhoods["cap"].apply(lambda x: ",".join(map(str, x)))
    # Write to processed file
    neighborhoods.to_file(out_file)
    return neighborhoods


def find_intersecting_cap(neighborhood_geometry, cap):
    cap_intersecting = []
    for id, cap in cap.iterrows():
        if (
            neighborhood_geometry.touches(cap.geometry) == False
        ):  # If they have at least one interior point in common
            if neighborhood_geometry.intersects(cap.geometry):
                cap_intersecting.append(id)
    return cap_intersecting


def buildings_data_processing(
    raw_file: str,
    fun_file: str,
    out_file: str,
    neighborhoods: gpd.GeoDataFrame,
    roads: nx.MultiDiGraph,
) -> None:
    buildings = gpd.read_file(raw_file)
    buildings_fun = dbfread.DBF(fun_file)
    buildings_fun = pd.DataFrame(buildings_fun)
    buildings_fun.rename(columns={"EDIFC_USO": "function"}, inplace=True)
    buildings_fun = buildings_fun.astype({"function": int})
    buildings = buildings.merge(buildings_fun, on="CLASSREF")
    buildings.drop("CLASSREF", axis=1, inplace=True)

    # Remove Transportation
    buildings = buildings[
        (buildings["function"] != 6)
        | ((buildings["function"] > 600) & (buildings["function"] < 605))  # 6 Transport
        | ((buildings["function"] > 60000) & (buildings["function"] < 60500))
    ]
    buildings.index.name = "id"

    # Analyze function and categorize buildings
    buildings_categorization(buildings)

    # Add neighborhood id to every building
    buildings["neighborhood"] = [
        find_neighborhood_by_pos(x, neighborhoods) for x in buildings.geometry
    ]

    # Save to file
    buildings.to_file(out_file)


def buildings_categorization(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["home"] = False
    buildings["day_act"] = False
    buildings["night_act"] = False

    buildings.loc[
        (buildings["function"] == 1) | (buildings["function"] == 101), "home"
    ] = True  # 1 or 101 Residenziale o abitativa,
    buildings.loc[
        (buildings["function"] != 8)
        & (buildings["function"] != 9)  # 8 Industrial, 9 Agricultural
        & ((buildings["function"] < 800) | (buildings["function"] > 905))
        | ((buildings["function"] < 80000) | (buildings["function"] > 80900)),
        "day_act",
    ] = True
    buildings.loc[
        (buildings["function"] == 1)
        | (  # 3012 Ospedale, 306 Forze dell'Ordine, 307 Vigili del fuoco
            buildings["function"] == 101
        )
        | (buildings["function"] == 3012)
        | (buildings["function"] == 306)
        | (buildings["function"] == 307),
        "night_act",
    ] = True
    return buildings


def find_neighborhood_by_pos(building_geometry, neighborhoods):
    intersections = [
        shapely.intersection(building_geometry, neighborhood)
        for neighborhood in neighborhoods.geometry
    ]
    neighborhoods["common_area"] = [intersection.area for intersection in intersections]
    return neighborhoods["common_area"].idxmax()

def public_transport_processing(
    roads: nx.MultiDiGraph,
    neighborhoods: gpd.GeoDataFrame,
    pedestrian_speed: int,
    out_file: str,
    **layers: str 
) -> None:
    for layer, path in layers.items():
        layer_df = gpd.read_file(os.path.join(directory, path))
        layer_df.to_crs('epsg:7791', inplace = True)
        layer_nodes = layer_df[layer_df.geom_type == 'Point'].reset_index(drop = True)
        layer_edges = layer_df[(layer_df.geom_type == 'LineString') | 
                               (layer_df.geom_type == "MultiLineString")]
        layer_graph = nx.MultiDiGraph(crs = "epsg:7791")
        
        #Selecting only nodes and edges inside the boundary of milan
        layer_nodes.loc[:, 'inside_milan'] = layer_nodes['geometry'].apply(lambda x: intersects(x, neighborhoods['geometry'])).any(axis = 1)
        layer_nodes = layer_nodes.loc[layer_nodes['inside_milan'] == True, :]
        layer_nodes.reset_index(drop = True, inplace = True)
    
        #Adding nodes to road graph
        layer_nodes['id'] = layer_nodes.index + len(roads.nodes) + 1
        layer_nodes.set_index('id', inplace = True)
        #TODO: Move Down 
        layer_graph.add_nodes_from([(i, {"x" : pos.x, "y" : pos.y, "transport" : f'{layer}'}) for i, pos in zip(layer_nodes.index, layer_nodes['geometry'])])

        #Converting original linestrings to edges
        def __segmentize(edge):
                if edge.geom_type == 'MultiLineString':
                    [__segmentize(line) for line in edge.geoms]
                elif edge.geom_type == 'LineString':
                    original_coords = list(edge.coords)
                    original_coords = array([[coord[0], coord[1]] for coord in original_coords])
                    layer_nodes.loc[:, 'positions'] = layer_nodes.loc[:, 'geometry'].apply(lambda node: [node.x, node.y])
                    layer_nodes.loc[:, 'nodes_in_edge'] = layer_nodes.loc[:, 'positions'].apply(lambda pos: intersects_xy(edge, pos[0], pos[1]))
                    nodes_in_edge = layer_nodes.loc[layer_nodes['nodes_in_edge'], :] 
                    if not nodes_in_edge.empty:
                        nodes_positions = array([[pos[0], pos[1]] for pos in nodes_in_edge['positions']])
                        distance = cdist(nodes_positions, original_coords)
                        nodes_in_edge.loc[:, 'nearest_coord_id'] = argmin(distance, axis = 1)
                        nodes_in_edge.sort_values(by = 'nearest_coord_id', inplace = True)
                        nodes_in_edge.reset_index(inplace = True)
                        nodes_in_edge.loc[:, 'previous_node'] = nodes_in_edge['id'].shift(1)
                        layer_graph.add_edges_from(zip(nodes_in_edge[1:]['id'], nodes_in_edge[1:]['previous_node']), transport = f'{layer}')
        
        layer_edges['geometry'].apply(lambda edge: __segmentize(edge))
                        
        #Connecting each station with road graph
        nearest_rode_node = [ox.distance.nearest_nodes(roads, node.x, node.y) for node in layer_nodes.geometry]
        road_to_station = list(zip(nearest_rode_node, list(layer_nodes.index)))
        station_to_road = list(zip(list(layer_nodes.index), nearest_rode_node))
        roads = nx.union(layer_graph, roads) #possibility of renaming
        roads.add_edges_from(road_to_station)
        roads.add_edges_from(station_to_road)
                
    #Simplyfing graph
    #TODO: Use conversion to gdf
    for node in roads.nodes:
        if 'osmid_original' in roads.nodes[node]:
            del roads.nodes[node]['osmid_original']

    #Adding edge speeds
    edges = ox.graph_to_gdfs(roads, nodes=False, edges=True)
    edges.loc[edges['transport'].isna(), 'speed_kph'] = pedestrian_speed
    edges.loc[edges['transport'] == 'subway', 'speed_kph'] = 65
    edges.loc[edges['transport'].isin(['tram', 'bus', 'trolleybus']), 'speed_kph'] = 35
    edges.loc[edges['length'].isna(), 'length'] = edges['geometry'].length
    edges.loc[:, "attributes_dict"] = ("{'speed_kph': " + edges["speed_kph"].astype(str) +
                                        ", 'length': " + edges['length'].astype(str) + "}")#TODO: transform in function
    edges.loc[:, "attributes_dict"] = edges.loc[:, "attributes_dict"].apply(literal_eval)
    attributes_dict = edges["attributes_dict"].to_dict()
    nx.set_edge_attributes(roads, attributes_dict)
    roads = ox.speed.add_edge_travel_times(roads)
    ox.io.save_graph_geopackage(roads,
                                filepath = os.path.join(directory, "data/processed/public_transport.gpkg"),
                                directed = True)

def main():
    #TODO: replace os.path.join with pathlib
    start_time = time.time()
    # Processing roads roads: --- 58.210232973098755 seconds ---
    roads_out_file = os.path.join(directory, r"data\processed\roads.gpkg")
    if os.path.isfile(roads_out_file) == True:
        roads_nodes = gpd.read_file(os.path.join(directory, r"data\processed\roads.gpkg"), layer = 'nodes').set_index('osmid')
        roads_edges = gpd.read_file(os.path.join(directory, r"data\processed\roads.gpkg"), layer = 'edges').set_index(['u', 'v', 'key'])
        roads = ox.graph_from_gdfs(roads_nodes, roads_edges)
    else:
        roads = obtaining_roads(
            city_name="Milan, Italy", crs="epsg:7791", out_file=roads_out_file
        )
    print("roads: " + "--- %s seconds ---" % (time.time() - start_time))

    # Processing neighborhoods neighborhoods: --- 0.7955336570739746 seconds ---
    start_time = time.time()
    neighborhoods_raw_file =  os.path.join(
        directory, r"data\raw\quartieri.geojson"
    )
    neighborhoods_pop_file = os.path.join(
        directory, r"data\raw\popolazione_quartieri.csv"
    )
    neighborhoods_income_file = os.path.join(
        directory, r"data\raw\redditi_cap_milano.csv"
    )
    neighborhoods_cap_file = os.path.join(
        directory, r"data\raw\cap_di_milano.geojson"
    )
    neighborhoods_out_file = os.path.join(
        directory, r"data\processed\neighborhoods.gpkg"
    )
    try:
        neighborhoods = gpd.read_file(neighborhoods_out_file)
        neighborhoods = neighborhoods.set_index("id")
    except DriverError:
        neighborhoods = neighborhoods_data_processing(
            neighborhoods_raw_file,
            neighborhoods_pop_file,
            neighborhoods_income_file,
            neighborhoods_cap_file,
            neighborhoods_out_file,
        )
    print("neighborhoods: " + "--- %s seconds ---" % (time.time() - start_time))
    
    # Processing public transport public_transport: --- 0.0 seconds ---
    start_time = time.time()
    public_transport_out_file = os.path.join(directory, r"data\processed\public_transport.graphml")
    if os.path.isfile(public_transport_out_file) == True:
        pass
    else:
        public_transport_processing(
            roads,
            neighborhoods,
            pedestrian_speed=5, #km/h
            out_file=public_transport_out_file,
            subway = r"data\raw\subway.geojson",
            tram = r"data\raw\tram.geojson",
            bus = r"data\raw\bus.geojson",
            trolleybus = r"data\raw\trolleybus.geojson",
        )
    print("public transport: " + "--- %s seconds ---" % (time.time() - start_time))
    
    # Processing buildings buildings: --- 861.1617505550385 seconds ---
    # TODO: Look into time, definitely too slow
    start_time = time.time() 
    buildings_raw_file = os.path.join(
        directory, r"data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp"
    )
    buildings_fun_file = os.path.join(
        directory, r"data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf"
    )
    buildings_out_file = os.path.join(directory, r"data\processed\buildings.shp")
    if os.path.isfile(buildings_out_file) == True:
        pass
    else:
        buildings_data_processing(
            buildings_raw_file,
            buildings_fun_file,
            buildings_out_file,
            neighborhoods,
            roads,
        )
    print("buildings: " + "--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()