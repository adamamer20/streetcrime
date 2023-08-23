import geopandas as gpd
import pandas as pd
import os.path
from shapely.geometry import Polygon, LineString
from fiona.errors import DriverError
import time
import os.path
import osmnx as ox
import networkx
import dbfread
import pyproj.crs

current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]
    current_directory = parent_directory

#TODO: CONVERT DATA PROCESSING TO A CLASS

def obtaining_roads(city_name : str,
                    crs : str,
                    out_file: str) -> networkx.MultiDiGraph:
    #Downloading the full dataset of roads in Milan
    roads = ox.graph_from_place(city_name) #simplifaction is already activated
    roads = ox.projection.project_graph(roads, to_crs=crs)
    roads = ox.simplification.consolidate_intersections(roads, tolerance = 25)    
    ox.io.save_graphml(roads, filepath=out_file)
    return roads
    
def neighborhoods_data_processing (raw_file : str,
                                   pop_file: str,
                                   income_file: str,
                                   cap_file: str,
                                   out_file: str) -> gpd.GeoDataFrame:
    #Read files
    neighborhoods_df = gpd.read_file(raw_file)
    neighborhoods_pop_df = pd.read_csv(pop_file)
    cap_income_df = pd.read_csv(income_file)
    cap_df = gpd.read_file(cap_file)
    #Drop unnecessary columns
    neighborhoods_df = neighborhoods_df[['ID_NIL', 'NIL', 'geometry']]
    neighborhoods_pop_df = neighborhoods_pop_df[['NIL', 'Totale', 'Totale_Milano']]
    cap_df = cap_df[['POSTCODE', 'geometry']]
    #Fix types
    cap_df['POSTCODE'] = cap_df['POSTCODE'].astype(int)
    #Compute proportion of population in each neighborhood
    neighborhoods_pop_df['prop'] = neighborhoods_pop_df['Totale'] / neighborhoods_pop_df['Totale_Milano'].astype(int)
    neighborhoods_pop_df = neighborhoods_pop_df[['NIL', 'prop']]
    #Rename columns to match other df
    neighborhoods_pop_df.rename(columns = {'NIL': 'ID_NIL'}, inplace = True)
    cap_income_df.rename(columns = {'CAP': 'POSTCODE', 'Reddito Medio' : 'income'}, inplace = True)
    #Merge dataframes
    neighborhoods_df = neighborhoods_df.merge(neighborhoods_pop_df, on = 'ID_NIL', how = 'left')
    cap_df = cap_df.merge(cap_income_df, on = 'POSTCODE', how = 'left')
    #Set indexes
    cap_df.set_index("POSTCODE", inplace = True)
    neighborhoods_df.set_index("ID_NIL", inplace = True)
    #Find in which neighborhoods are cap located
    neighborhoods_df['cap'] = None
    for id, neighborhood in neighborhoods_df.iterrows(): #do it with list comprehension (look at set building entrance)
        neighborhoods_df.at[id, 'cap'] = find_intersecting_cap(neighborhood.geometry, cap_df)
    #Rename columns
    neighborhoods_df.rename(columns = {'NIL': 'name'}, inplace = True)
    neighborhoods_df.rename_axis('id', inplace = True)
    #Add average income
    for id, neighborhood in neighborhoods_df.iterrows():
        sum_income = 0
        n = 0
        for cap in neighborhood.cap:
             sum_income += cap_df.at[cap, 'income']
             n += 1
        neighborhoods_df.at[id, 'income'] = sum_income / n
    #Change CRS to epsg:7791
    neighborhoods_df.to_crs("epsg:7791", inplace = True)
    #Convert CAP list to string (to save the file)
    neighborhoods_df['cap'] = neighborhoods_df['cap'].apply(lambda x: ','.join(map(str, x)))
    #Write to processed file
    neighborhoods_df.to_file(out_file)
    return neighborhoods_df

def find_intersecting_cap(neighborhood_geometry, cap_df):
    cap_intersecting = []
    for id, cap in cap_df.iterrows():
        if neighborhood_geometry.touches(cap.geometry) == False: #If they have at least one interior point in common
            if neighborhood_geometry.intersects(cap.geometry):
                cap_intersecting.append(id)
    return cap_intersecting
    
def buildings_data_processing(raw_file : str, 
                              fun_file : str,
                              out_file : str,
                              neighborhoods_df : gpd.GeoDataFrame,
                              roads: networkx.MultiDiGraph) -> None:
    buildings_df = gpd.read_file(raw_file)
    buildings_fun_dbf = dbfread.DBF(fun_file)
    buildings_fun_df = pd.DataFrame(buildings_fun_dbf)
    buildings_fun_df.rename(columns = {'EDIFC_USO': 'function' }, inplace = True)
    buildings_fun_df = buildings_fun_df.astype({"function": int})
    buildings_df = buildings_df.merge(buildings_fun_df, on = "CLASSREF")
    buildings_df.drop("CLASSREF", axis=1, inplace=True)
    
    #Sample only 50 buildings
    buildings_df = buildings_df.sample(n=4, random_state=1)
    
    #Remove Transportation 
    buildings_df = buildings_df[(buildings_df['function'] != 6) | #6 Transport
                                ((buildings_df['function'] > 600) & (buildings_df['function'] < 605)) |
                                ((buildings_df['function']>60000) & (buildings_df['function'] < 60500))    
                                ]
    buildings_df.index.name = "id"
    
    #Analyze function and categorize buildings
    buildings_categorization(buildings_df)
    
    #Add neighborhood id to every building
    buildings_df['neighborhood'] = None
    for id, building in buildings_df.iterrows(): #do it with list comprehension (look at set building entrance)
        buildings_df.at[id, 'neighborhood'] = find_containing_neighborhood(building.geometry, neighborhoods_df)
    
    #Add entrance node for every building
    set_building_entrance(buildings_df, roads)
    
    #Save to file
    buildings_df.to_file(out_file)

def buildings_categorization(buildings_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings_df['home'] = False
    buildings_df['day_act'] = False
    buildings_df['night_act'] = False
    
    buildings_df.loc[(buildings_df['function'] == 1) |
                     (buildings_df['function'] == 101) , 'home'] = True #1 or 101 Residenziale o abitativa,
    buildings_df.loc[(buildings_df['function'] != 8) & #8 Industrial, 9 Agricultural
                     (buildings_df['function'] != 9) & 
                                ((buildings_df['function'] < 800) | (buildings_df['function'] > 905)) |
                                ((buildings_df['function'] < 80000) | (buildings_df['function'] > 80900))    
                                , 'day_act'] = True
    buildings_df.loc[(buildings_df['function'] == 1) | #3012 Ospedale, 306 Forze dell'Ordine, 307 Vigili del fuoco
                     (buildings_df['function'] == 101) |
                     (buildings_df['function'] == 3012) | 
                     (buildings_df['function'] == 306) | 
                     (buildings_df['function'] == 307),  
                     'night_act'] = True
    return buildings_df
   
def find_containing_neighborhood(building_geometry, neighborhoods_df):
    for id, neighborhood in neighborhoods_df.iterrows():
        if neighborhood.geometry.contains(building_geometry):
            return id
    
#Adding entrance node for every building
def set_building_entrance(buildings_df: gpd.GeoDataFrame, 
                          roads: networkx.MultiDiGraph) -> None:
    entrance_nodes = [ox.distance.nearest_nodes(roads, x, y) for x, y in zip(buildings_df.centroid.x, buildings_df.centroid.y)]
    buildings_df['entrance_node'] = entrance_nodes
    return buildings_df

def main():
    start_time = time.time()
    #Processing roads roads: --- 58.210232973098755 seconds ---
    roads_out_file = os.path.join(parent_directory, r"data\processed\roads.graphml") 
    if os.path.isfile(roads_out_file) == True:
        roads = ox.io.load_graphml(roads_out_file)
    else:
        roads = obtaining_roads(city_name = "Milan, Italy", 
                                crs = "epsg:7791", 
                                out_file = roads_out_file)
    print("roads: " + "--- %s seconds ---" % (time.time() - start_time))

    #Processing neighborhoods neighborhoods: --- 0.7955336570739746 seconds ---
    start_time = time.time()
    neighborhoods_raw_file = os.path.join(parent_directory, r"data\raw\quartieri.geojson") 
    neighborhoods_pop_file = os.path.join(parent_directory, r"data\raw\popolazione_quartieri.csv")
    neighborhoods_income_file =os.path.join(parent_directory, r"data\raw\redditi_cap_milano.csv")
    neighborhoods_cap_file = os.path.join(parent_directory, r"data\raw\cap_di_milano.geojson")
    neighborhoods_out_file = os.path.join(parent_directory, r"data\processed\neighborhoods.shp")
    try:
        neighborhoods_df = gpd.read_file(neighborhoods_out_file)
    except DriverError:
        neighborhoods_df = neighborhoods_data_processing(neighborhoods_raw_file, neighborhoods_pop_file, 
                                                         neighborhoods_income_file, neighborhoods_cap_file, neighborhoods_out_file)
    neighborhoods_df = neighborhoods_df.set_index('id')
    print("neighborhoods: " + "--- %s seconds ---" % (time.time() - start_time))

    #Processing buildings buildings: --- 71.95271301269531 seconds ---
    start_time = time.time()
    buildings_raw_file = os.path.join(parent_directory, r"data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp")
    buildings_fun_file = os.path.join(parent_directory, r"data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf")
    buildings_out_file = os.path.join(parent_directory, r"data\processed\buildings.shp")
    if os.path.isfile(buildings_out_file) == True:
        pass
    else:
        buildings_data_processing(buildings_raw_file, buildings_fun_file, 
                                  buildings_out_file, neighborhoods_df, roads)
    print("buildings: " + "--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
    