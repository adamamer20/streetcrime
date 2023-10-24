from ast import literal_eval
import geopandas as gpd
import pandas as pd
import os.path
import shapely
from shapely import intersection
import time
import os.path
import osmnx as ox
import networkx as nx
import dbfread
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy import array, argmin
from warnings import warn
from pyproj import crs
import mesa_geo as mg

#TODO: FIX CLASS AND MOVE TO EXAMPLES
class Milan_data_processing:
    def __init__(self, crs: str):
        self.crs = crs
        
    def neighborhoods_processing(
        self,
        raw_file: str, pop_file: str, income_file: str, cap_file: str, out_file: str
    ) -> gpd.GeoDataFrame:
        
        start_time = time.time()
        
        if os.path.isfile(out_file):
            print(f"Neighborhoods file already in {out_file}, loading from file")
            neighborhoods = gpd.read_file(out_file).set_index("id")
        else:
            print(f"Neighborhoods file not present in {out_file}, starting processing")
            
            # Read files
            neighborhoods = gpd.read_file(raw_file).set_index("ID_NIL")[["NIL", "geometry"]]
            neighborhoods_pop = pd.read_csv(pop_file).set_index("NIL")[["Totale", "Totale_Milano"]].rename_axis('ID_NIL')
            cap_income = pd.read_csv(income_file).set_index('CAP').rename_axis('POSTCODE')
            cap = gpd.read_file(cap_file).set_index("POSTCODE")[["geometry"]]
            
            # Fix types
            cap.index = cap.index.astype(int)
            
            # Compute proportion of population in each neighborhood
            neighborhoods_pop["prop"] = neighborhoods_pop["Totale"] / neighborhoods_pop[
                "Totale_Milano"
            ].astype(int)
            
            neighborhoods_pop = neighborhoods_pop["prop"]
            
            # Merge dataframes
            neighborhoods = neighborhoods.merge(neighborhoods_pop, left_index=True, right_index=True, how="left")
            
            cap = cap.merge(cap_income, left_index=True, right_index=True, how="left")
            
            # Find in which neighborhoods are cap located
            neighborhoods_caps = neighborhoods.geometry.apply(lambda x: (intersection(x, cap.geometry).area/x.area))

            # Generate data for skewnorm per neighborhood
            frequency_columns = [
                column for column in cap_income.columns.str.split(" ") if "Frequenza" in column
            ]
            
            frequency_columns = [" ".join(column) for column in frequency_columns]
            
            mean_columns = [
                column for column in cap_income.columns.str.split(" ") if "Media" in column
            ]
            
            mean_columns = [" ".join(column) for column in mean_columns]
            
            
            city_data = np.array([])
            
            for id, _ in neighborhoods.iterrows():
                #Selecting caps per neighborhoods where overlap is greater than 0
                neighborhood_caps = neighborhoods_caps.loc[id, neighborhoods_caps.loc[id] > 0].to_frame('overlap')
                
                #Merge with data on income cap
                neighborhood_caps = neighborhood_caps.join(cap_income)
                
                #Compute frequency and average per neighborhood
                neighborhood_caps[frequency_columns] = neighborhood_caps[frequency_columns].multiply(neighborhood_caps['overlap'], axis = 0)
                neighborhood_caps[mean_columns] = neighborhood_caps[mean_columns].multiply(neighborhood_caps['overlap'], axis = 0)
                neighborhood_df = neighborhood_caps.sum()
                neighborhood_df = neighborhood_df/neighborhood_df['overlap']
                neighborhood_df = neighborhood_df.to_frame(id).T.drop('overlap', axis = 1)
                
                #Generate data for skewnorm
                neighborhood_df['sum_income'] = 0
                neighborhood_data = np.array([])
                for i in range(len(frequency_columns)):
                    neighborhood_data = np.concatenate((neighborhood_data, np.full(int(neighborhood_df[frequency_columns[i]].loc[id]), neighborhood_df[mean_columns[i]])))
                    neighborhood_df['sum_income'] += neighborhood_df[mean_columns[i]] * neighborhood_df[frequency_columns[i]]
                city_data = np.concatenate((city_data, neighborhood_data))
                ae, loce, scalee = scipy.stats.skewnorm.fit(neighborhood_data)
                neighborhoods.at[id, "ae"] = ae
                neighborhoods.at[id, "loce"] = loce
                neighborhoods.at[id, "scalee"] = scalee
                neighborhoods.at[id, "sum_income"] = neighborhood_df['sum_income'].loc[id]
                neighborhoods.at[id, "sum_frequency"] = neighborhood_df[frequency_columns].sum(axis = 1).loc[id]
                
            city_ae, city_loce, city_scalee = scipy.stats.skewnorm.fit(city_data)
            neighborhoods["city_ae"] = city_ae
            neighborhoods["city_loce"] = city_loce
            neighborhoods["city_scalee"] = city_scalee
            neighborhoods['mean_income'] = neighborhoods['sum_income'] / neighborhoods['sum_frequency']
            neighborhoods['city_mean_income'] = neighborhoods['sum_income'].sum() / neighborhoods['sum_frequency'].sum()

            # Drop unnecessary columns
            neighborhoods.drop(['sum_income', 'sum_frequency'], axis = 1, inplace = True)

            # Rename columns
            neighborhoods.rename(columns={"NIL": "name"}, inplace=True)
            neighborhoods.rename_axis("id", inplace=True)

            # Change self.crs to epsg:7791
            neighborhoods.to_crs("epsg:7791", inplace=True)
            # Write to processed file
            neighborhoods.to_file(out_file)
        
        print("Processed neighborhoods: " + "--- %s seconds ---" % (time.time() - start_time))            
                    
    def buildings_processing(
        self,
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

    def buildings_categorization(
        self,
        buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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

    def public_transport_processing(
        self,
        pedestrian_speed: int,
        out_file: str,
        **layers: str 
    ) -> None:
        for layer, path in layers.items():
            layer_df = gpd.read_file(path)
            layer_df.to_crs('epsg:7791', inplace = True)
            layer_nodes = layer_df[layer_df.geom_type == 'Point'].reset_index(drop = True)
            layer_edges = layer_df[(layer_df.geom_type == 'LineString') | 
                                (layer_df.geom_type == "MultiLineString")]
            layer_graph = nx.MultiDiGraph(crs = self.crs)
            
            #Selecting only nodes and edges inside the boundary of milan
            layer_nodes.loc[:, 'inside_milan'] = layer_nodes['geometry'].apply(lambda x: intersects(x, neighborhoods['geometry'])).any(axis = 1)
            layer_nodes = layer_nodes.loc[layer_nodes['inside_milan'], :]
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
                                    filepath = "data/processed/public_transport.gpkg",
                                    directed = True)

def main():
    Milan = Milan_data_processing(crs = "epsg:7991")
    Milan.neighborhoods_processing(raw_file = "data/raw/quartieri.geojson",
                                  pop_file = "data/raw/popolazione_quartieri.csv",
                                  income_file = "data/raw/redditi_cap_milano.csv",
                                  cap_file = "data/raw/cap_di_milano.geojson",
                                  out_file = "data/processed/neighborhoods.gpkg")
    
    '''Milan.public_transport_processing(
            roads,
            neighborhoods,
            pedestrian_speed=5, #km/h
            out_file=public_transport_out_file,
            subway = "data/raw/subway.geojson",
            tram = "data/raw/tram.geojson",
            bus = "data/raw/bus.geojson",
            trolleybus = "data/raw/trolleybus.geojson",
        )
    print("public transport: " + "--- %s seconds ---" % (time.time() - start_time))
    
    buildings_raw_file = "data/raw/DBT_2020/DBT 2020 - SHAPE/EDIFC_CR_EDF_IS.shp"
    buildings_fun_file ="data/raw/DBT_2020/DBT 2020 - SHAPE/EDIFC_EDIFC_USO.dbf"
    buildings_out_file = "data/processed/buildings.shp"
    Milan.buildings_processing(
            buildings_raw_file,
            buildings_fun_file,
            buildings_out_file,
            neighborhoods,
            roads,
        )
    print("buildings: " + "--- %s seconds ---" % (time.time() - start_time))'''
        
if __name__ == "__main__":
    main()