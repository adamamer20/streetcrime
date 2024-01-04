from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point
import os.path
from mesa_frames.space import GeoSpace
from streetcrime.agents.mover import Mover
from streetcrime.agents.worker import Worker
from streetcrime.agents.criminal import Criminal
import networkx as nx
import osmnx as ox
import pickle
import random
import ast
from typing import Callable
from warnings import warn
from fiona.errors import DriverError
from streetcrime.utility import _method_parser
from ast import literal_eval
import pandas as pd
from scipy.stats import skewnorm
import time # To measure the time of the loading of the files
import overpy
from pyproj import CRS
from mesa_frames.agent import Agent

class City(GeoSpace):
    """
    The City class is the GeoSpace in which agents move.
    It is used to store the road & public transport network, the buildings and neighborhoods dataframes.
    
    Parameters:
    ----------
        city_name : str, optional 
            City to pass to `osmnx` to download layers. Optional if there is not a layer which call an obtaining function
        model : mesa.Model
            The model in which the city is used
        crs : pyproj.CRS
            The crs of the city
        **layers : str | (Callable, list[str]) | (Callable, dict[str, str]
            The layers of the city. Can be 'roads', 'public_transport', 'neighborhoods' or 'buildings'
            If str, it is the path to the file containing the layer
            If (Callable, dict[str, str]) it is a method of city with keyword parameters of the associated function
            If (Callable, list[str]) it is a method of city with ordered parameters of the associated function
        
    Methods:
    ----------
        _obtain_roads(out_file : str, tolerance : int = 20, traffic_factor : int = 1) -> nx.MultiDiGraph
            It reads a .gpkg or .graphml file specified in `out_file` or obtains the road network from `osmnx.graph_from_place`
        get_random_building(agent : agent, function : str) -> int
            Returns a random building id based on the function passed as argument
        find_neighborhood_by_pos(position : Point) -> int
            Returns the neighborhood id of the neighborhood in which the position passed as argument is contained
        self.buildings.geometry.distance(position : Point) -> gpd.GeoSeries
            Returns the distance from the position passed as argument to all the buildings in the city
        update_information() -> None
            Updates 'yesterday_crimes' and 'yesterday_visits' columns of the buildings dataframe with data
            collected in 'today_crimes' and 'today_visits' columns of the neighborhoods dataframe
    """
    
    def __init__(self, 
                 crs: CRS,
                 city_name : str,
                 **layers : str) -> None:
        """
            Parameters:

            crs(str)
                The crs of the city. Should be a projected crs and not a geographic one. 
        """
            
        super().__init__(crs=crs)
        self.city_name = city_name
        self.roads_nodes = None
        self.roads_edges = None
        self.roads = None

        # Read paths files
        if os.path.isfile(f"outputs/city/{self.city_name.split(',')[0]}_paths.csv"):
            self.paths = pd.read_csv(f"outputs/city/{self.city_name.split(',')[0]}_paths.csv")
            #need to read strings as list types for .explode(), faster than using ast.literal_eval
            self.paths.path = self.paths.path.str.strip('[]').str.split(', ')
        else:
            self.paths = pd.DataFrame(columns = ['node', 'destination', 'path'])
            
        # If additonal layers are specified
        if len(layers) > 0:
            print("Loading layers : ...")
            for layer, arg in layers.items():
                if os.path.isfile(arg):
                    layer_time = time.time()
                    print(f"Loading {layer} from {arg}")
                    match layer:
                        case "neighborhoods":
                            self.neighborhoods = gpd.read_file(arg).set_index('id')
                            self.income_distribution = skewnorm(a = self.neighborhoods['city_ae'].iloc[0], 
                                                                loc = self.neighborhoods['city_loce'].iloc[0], 
                                                                scale = self.neighborhoods['city_scalee'].iloc[0]) 
                            self.neighborhoods.drop(columns = ['city_ae', 'city_loce', 'city_scalee'], inplace = True)
                        case _:
                            setattr(self, layer, gpd.read_file(arg))
                    print(f"Loaded {layer}: " + "--- %s seconds ---" % (time.time() - layer_time))
                else:
                    setattr(self, layer, _method_parser(self, arg))
        
        
    #TODO: remove nodes without outgoing edges
    def load_data(self,
                  tolerance : int = 15,
                  traffic_factor : int = 1,
                  roads_file : str = None, #TODO: specify default value
                  buildings_file : str = None,
                  building_categories : pd.DataFrame = None) -> None:
        """Obtains the data of the city from OSMNX and saves it in the specified files
        or reads data from the file passed as arguments"""

        #Check parameters
        if not 0 < traffic_factor  <= 1:
            raise ValueError(f"Traffic factor set to {traffic_factor}. Should be between 0 and 1")
        if tolerance <= 0:
            raise ValueError(f"Tolerance set to {tolerance}. Should be greater than 0")
        if not os.path.isdir("outputs"):
            os.mkdir("outputs")
        if roads_file:
            roads_file = f"outputs/city/{roads_file}"
        else:
            roads_file = f"outputs/city/{self.city_name.split(',')[0]}_roads.gpkg"
            
        #Obtain roads
        if os.path.isfile(roads_file):
            print(f"Roads already downloaded in {roads_file}. Loading...")
            if ".gpkg" in roads_file:
                #TODO: remove useless columns
                self.roads_nodes = gpd.read_file(roads_file, layer = 'nodes').set_index('osmid')
                self.roads_edges = gpd.read_file(roads_file, layer = 'edges').set_index(['u', 'v', 'key'])
                self.roads = nx.DiGraph(ox.graph_from_gdfs(self.roads_nodes, self.roads_edges))                
                self.roads_edges = self.roads_edges.sort_values('travel_time').reset_index().drop(columns = ['key']).groupby(['u', 'v']).first()
                return
            
        roads = self._obtain_roads(tolerance, traffic_factor)
        
        #Obtain buildings
        buildings = self._obtain_buildings(buildings_file, building_categories)
        buildings['nearest_node'] = buildings.geometry.apply(lambda x: ox.nearest_nodes(roads, x.centroid.x, x.centroid.y))
        buildings = buildings.groupby('nearest_node').count()
        buildings = buildings[['home', 'work', 'activity', 'open_day', 'open_night']]
        nx.set_node_attributes(roads, buildings.transpose().to_dict())
        self.roads_nodes = ox.graph_to_gdfs(roads, nodes=True, edges=False)
        ox.save_graph_geopackage(roads, filepath = roads_file, directed = True)
        print(f"Saved roads in {roads_file}")
        self.roads = nx.DiGraph(roads)
        
        #Compute all pairs shortest path (faster than lazy computation at each iteration)
        
    
    def get_random_nodes(self, 
                        function: str = None,
                        time: str = None,
                        agent : Agent = None, 
                        decision_rule : str = None,
                        n = 1) -> int:
                
        weights = None
        
        if function and time: 
            weights = self.roads_nodes[function]*self.roads_nodes[time]
        elif function:
            weights = self.roads_nodes[function]
        
        #if decision_rule:
        #    node = self.roads_nodes.sample(n = n)

        if n == 1:
            return self.roads_nodes.sample(n = 1, weights = weights).index[0]
        else:  
            return self.roads_nodes.sample(n = n, weights = weights, replace = True).index
                        
    def _obtain_roads(self,
                      tolerance : int = 15,
                      traffic_factor : int = 1) -> nx.MultiDiGraph:
        """Download roads using `osmnx.graph_from_place`

        Parameters
        ----------
        tolerance : int, default = 15
            The tolerance according to which intersectation are consolidated with `osmnx.simplification.consolidate_intersections`.
        traffic_factor : int, default = 1
            The factor with which speeds will be refactored. If not specified, velocities will be set to max speed per road.


        Returns
        -------
        `networkx.MultiDiGraph`
            returns a networkx Multi Directed Graph

        Raises
        ------
        `NameError`
            If `out_file` does not end in .graphml or .gpkg
        `ValueError`
            If `self.city_name` is not specified and file in `out_file` is not present
        """
        start_time = time.time()
        print(f"Downloading roads of {self.city_name}...")
                    
        # Downloading the full dataset of roads
        roads = ox.graph_from_place(self.city_name)  # simplification is already activated
        roads = ox.projection.project_graph(roads, to_crs=self.crs)
            
        if tolerance > 0:
            roads = ox.simplification.consolidate_intersections(roads, tolerance=tolerance)

        roads = ox.speed.add_edge_speeds(roads)
        nodes, edges = ox.graph_to_gdfs(roads, nodes=True, edges=True)
        
        # Calculate the speed of each edge
        if traffic_factor != 1:
            edges["speed_kph"] = edges["speed_kph"]*traffic_factor
            roads = ox.graph_from_gdfs(nodes, edges)
        
        # Add travel times
        roads = ox.speed.add_edge_travel_times(roads)

        #Remove nodes without outgoing edges
        out_degree = pd.Series(dict(roads.out_degree))
        roads.remove_nodes_from(out_degree[out_degree == 0].index)

        # Find all pairs shortest paths (faster than lazy computation at each iteration)
        paths = dict(nx.all_pairs_dijkstra_path(roads, weight = 'travel_time'))
        data = []
        # Iterate through the nested dictionary
        for origin, destinations in paths.items():
            for destination, path in destinations.items():
                # Append a tuple with the origin, destination, and path to the list
                data.append((origin, destination, path))
        paths = pd.DataFrame(data, columns=['node', 'destination', 'path'])
        paths.astype({'node': 'int64', 'destination': 'float64'})
        paths.to_csv(f"outputs/city/{self.city_name.split(',')[0]}_paths.csv", index = False)
        self.paths = paths

        print("Downloaded roads: " + "--- %s seconds ---" % (time.time() - start_time))
        return roads
    
    def _obtain_buildings(self, buildings_file: str = None, building_categories : pd.DataFrame = None) -> gpd.GeoDataFrame:
        start_time = time.time()
        
        if buildings_file:
            buildings_file = f"outputs/{buildings_file}"
        else:
            buildings_file = f"outputs/{self.city_name.split(',')[0]}_buildings.gpkg"

        print(f"Downloading buildings of {self.city_name}...")
        buildings = ox.features_from_place(self.city_name, 
                                           tags = {'building': True})
        print('Downloaded buildings: ' + "--- %s seconds ---" % (time.time() - start_time))
        buildings.to_crs(self.crs, inplace = True)
        buildings.reset_index(inplace = True)
        buildings = buildings[['geometry', 'building']]
        #categorize buildings
        if not building_categories:
            building_categories = [
                ["apartments", True, None, None, True, None],
                ["barracks", True, None, None, True, None],
                ["bungalow", True, None, None, True, True],
                ["cabin", True, None, True, True, True],
                ["detached", True, None, None, True, True],
                ["dormitory", True, None, None, True, True],
                ["farm", True, True, None, True, True],
                ["ger", True, None, None, True, True],
                ["hotel", None, True, True, True, True],
                ["house", True, None, None, True, True],
                ["houseboat", True, None, True, True, True],
                ["residential", True, None, None, True, True],
                ["semidetached_house", True, None, None, True, True],
                ["static_caravan", True, None, None, True, True],
                ["stilt_house", True, None, True, True, True],
                ["terrace", True, None, None, True, True],
                ["tree_house", True, None, True, True, None],
                ["trullo", True, None, None, True, True],
                ["commercial", None, True, True, True, None],
                ["industrial", None, True, None, True, None],
                ["kiosk", None, True, True, True, True],
                ["office", None, True, None, True, None],
                ["retail", None, True, True, True, True],
                ["supermarket", None, True, True, True, True],
                ["warehouse", None, True, None, True, None],
                ["church", None, True, True, True, None],
                ["mosque", None, True, True, True, None],
                ["synagogue", None, True, True, True, None],
                ["temple", None, True, True, True, None],
                ["school", None, True, None, True, None],
                ["university", None, True, None, True, None],
                ["hospital", None, True, True, True, True],
                ["fire_station", None, True, None, True, True],
                ["government", None, True, None, True, None],
                ["yes", True, True, True, True, True]
            ]
            columns = ["building", "home", "work", "activity", "open_day", "open_night"]
            building_categories = pd.DataFrame(building_categories, columns=columns)
        buildings = buildings.merge(building_categories, on = 'building', how = 'left')
        if buildings_file:
            buildings.to_file(buildings_file)
            print(f"Saved buildings in {buildings_file}")
        return buildings
              
    #TODO: implement public transport retrieval without passing through overpass turbo
    def obtaining_public_transport(self):
        # Load JSON data from Overpass API
        data = json.load(your_json_data)

        # Process relations
        for relation in data['elements']:
            if relation['type'] == 'relation':
                # Initialize an empty list to store line strings
                line_strings = []
                for member in relation['members']:
                    if member['type'] == 'way':
                        points = [(node['lon'], node['lat']) for node in member['geometry']]
                        line_strings.append(LineString(points))

                # Combine line strings into a single geometry
                # This could be a MultiLineString or another appropriate geometry type
                relation_geometry = combine_line_strings(line_strings)

                # Do something with the relation geometry

    ''' deprecated by get_random_nodes()
    def get_random_building(self, 
                            function: str = None,
                            agent : mesa.Agent = None,
                            decision_rule : str = None)-> int:
        """This method act on the City.buildings_df to obtain a random building id based on the function and the agent that requested it.
        Args:
            agent (agent, optional): The agent that is looking for a building, by default None.
            function (str, optional): The function of the building that the agent is looking for, by default None. Can be "home", "work", "day_act" or "night_act".  
        
        Returns:
            int: The id of the building
        
        """
        #TODO: maybe weights for home should also be in the buildings df?
        if decision_rule is None:
            _building = self.buildings[self.buildings['home'] == 1].sample(n = 1)
        else:
            weights = self._weights_parser(agent, decision_rule)
            if function == "home":
                #_neighborhood = self.neighborhoods.sample(n=1, weights=weights)
                # _building = self.buildings[(self.buildings['home']) & (self.buildings['neighborhood'] == _neighborhood.index[0])].sample(n = 1)
                _building = self.buildings[self.buildings['home'] == 1].sample(n = 1)
            else:
                _building = self.buildings[self.buildings[function] == 1].sample(n=1, weights=weights)
        return _building.index[0]'''

    ''' #TODO: implement with DF
      def find_neighborhood_by_pos(self, position: Point) -> int:
        """Find the neighborhood in which the position passed as argument is contained

        Parameters:
        ----------
            position (Point): The position to find the containing neighborhood

        Returns:
        ----------
            int: The id of the neighborhood
        """
        try:
            intersecting_neighborhoods = self.neighborhoods[self.neighborhoods.geometry.contains(position)]
            index = intersecting_neighborhoods.index[0]
        except IndexError:
            warn(f"Position {position} is not contained in any neighborhood", RuntimeWarning)
            index = None
        return index'''

    ''' #TODO: ditch paths altogether?
    def paths_path(
        self,
        network : str,
        source_node: int,
        target_node: int,
        path: list[int],
    ) -> None:
        #print(f"caching path... current number of pathsd paths: {len(self._path_paths)}")
        paths = getattr(self, f"_paths_{network}")
        paths[(source_node, target_node)] = path      

    def get_pathsd_path(
        self, 
        network: str,
        source_node: int,
        target_node: int
    ) -> list[int]:
        paths = getattr(self, f"_paths_{network}")
        return paths.get((source_node, target_node), None)
    
    def _load_paths_files(self, layers : list[str]) -> None:
        for layer in layers:
            try:
                with open(f"/outputs/_paths/_paths_{layer}.pkl", "rb") as paths_file: 
                    setattr(self, f"_paths_{layer}", pickle.load(paths_file))
            except (FileNotFoundError, EOFError):
                setattr(self, f"_paths_{layer}", dict())
                warn(f"/outputs/_paths/_paths_{layer}.pkl not found, creating a new file", RuntimeWarning)

    def save_paths_files(self, layers : list[str]) -> None:
        for layer in layers:
            with open(f"/outputs/_paths/_paths_{layer}.pkl", "wb") as paths_file:
                pickle.dump(getattr(self, f"_paths_{layer}"), paths_file)'''
    
    '''#TODO: deprecated by current implementation
    def _weights_parser(self, agent: mesa.Agent, decision_rule : str):
        decision_rule = decision_rule.split(",")
        df = getattr(self, decision_rule[0]) 
        df = pd.eval(decision_rule[1].strip(), target=df)        
        return df.weights.replace(np.inf, 0)'''

