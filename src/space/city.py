from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import mesa 
import mesa_geo as mg
import shapely
from shapely.geometry import Point
import os.path
from src.agents.mover import Mover
from src.agents.worker import Worker
from src.agents.criminal import Criminal
import networkx as nx
import osmnx as ox
import pickle
import random
import ast
from typing import Callable
from warnings import warn
from fiona.errors import DriverError
from src.utility import _method_parser
from ast import literal_eval
import pandas as pd
from scipy.stats import skewnorm
import time # To measure the time of the loading of the files

class City(mg.GeoSpace):
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
        obtaining_roads(out_file : str, tolerance : int = 20, traffic_factor : int = 1) -> nx.MultiDiGraph
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
                crs: str,
                city_name : str = None,
                **layers : str) -> None:
        """
            Arguments:
            crs(str)
                The crs of the city
            roads(networkx.MultiDiGraph, optional)
                The road network of the city. Will be converted to a RoadNetwork object (see src/space/roads)
            neighborhoods(gpd.GeoDataFrame, optional)
                The neighborhoods dataframe of the city
            buildings_df(gpd.GeoDataFrame, optional)
                The buildings dataframe of the city
        """
        city_time = time.time()
        super().__init__(crs=crs)
        self.city_name = city_name
        print("Loading city : ...")
        for layer, arg in layers.items():
            if os.path.isfile(arg):
                layer_time = time.time()
                print(f"Loading {layer} from {arg}")
                match layer:
                    case "roads" | "public_transport":
                        if ".gpkg" in arg:
                            setattr(self, f'{layer}_nodes', gpd.read_file(arg, layer = 'nodes').set_index('osmid'))
                            setattr(self, f'{layer}_edges', gpd.read_file(arg, layer = 'edges').set_index(['u', 'v', 'key']))
                            setattr(self, layer, ox.graph_from_gdfs(getattr(self, f'{layer}_nodes'), getattr(self, f'{layer}_edges'))) 
                        elif '.graphml' in arg:
                            setattr(self, layer, ox.load_graphml(arg))
                    case "public_transport":
                        self.public_transport = ox.load_graphml(arg)
                    case "buildings":
                        self.buildings = gpd.read_file(arg).rename(columns = {'neighborho': 'neighborhood'}).set_index('id')
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
        
        self._load_cache_files(['roads', 'public_transport'])
        print("Loaded city: " + "--- %s seconds ---" % (time.time() - city_time))

    def obtaining_roads(self,
                        out_file: str,
                        tolerance : int = 20,
                        traffic_factor : int = 1, 
                        ) -> nx.MultiDiGraph:
        """It reads a .gpkg or .graphml file specified in `out_file` or obtains the road network from `osmnx.graph_from_place`.

        Parameters
        ----------
        out_file : str
            Final location of the downloaded roads. If a file is already present, it will be loaded.
        tolerance : int, default = 20
            The tolerance according to which intersectation are consolidated with `osmnx.simplification.consolidate_intersections`.¡
        traffic_factor : int, default = 1
            The factor with which speeds will be refactored. If not specified, not used.


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
        
    #Check parameters validity
        if traffic_factor > 1:
            warn(f"Traffic factor set to {traffic_factor}. Should be less than 1")
        if not ((".graphml" in out_file) | (".gpkg" in out_file)):
            raise NameError("out_file should end in .gpkg or .graphml")   
        if os.path.isfile(out_file):
            print(f"Roads already downloaded in {out_file}. Loading...")
            if ".gpkg" in out_file:
                            roads_nodes = gpd.read_file(out_file, layer = 'nodes').set_index('osmid')
                            roads_edges = gpd.read_file(out_file, layer = 'edges').set_index(['u', 'v', 'key'])
                            roads = ox.graph_from_gdfs(roads_nodes, roads_edges)
            else:
                roads = ox.load_graphml(out_file)
        else:  
            # Downloading the full dataset of roads
            roads = ox.graph_from_place(self.city_name)  # simplifaction is already activated
            if not self.crs:
                roads = ox.projection.project_graph(roads, to_crs=self.crs)
            roads = ox.simplification.consolidate_intersections(roads, tolerance=tolerance)
            roads = ox.speed.add_edge_speeds(roads)
            edges = ox.graph_to_gdfs(roads, nodes=False, edges=True)
            if not traffic_factor == 1:
                edges["speed_kph"] = edges["speed_kph"]*traffic_factor
                edges["attributes_dict"] = "{'speed_kph': " + edges["speed_kph"].astype(str) + "}"
                edges["attributes_dict"] = edges["attributes_dict"].apply(literal_eval)
                attributes_dict = edges["attributes_dict"].to_dict()
                nx.set_edge_attributes(roads, attributes_dict)
            roads = ox.speed.add_edge_travel_times(roads)
            if ".graphml" in out_file:
                ox.io.save_graphml(roads, filepath=out_file)
            elif ".gpkg" in out_file:
                ox.io.save_graph_geopackage(roads, filepath = out_file, directed = True)
        
            print("Loaded roads: " + "--- %s seconds ---" % (time.time() - start_time))
        return roads
    
    #TODO: implement buildings retrieval with osmnx
    def obtaining_buildings(self):
        ox.features.features_from_place(self.city_name, tags=None, which_result=None, buffer_dist=None)

    #TODO: implement public transport retrieval without passing to overpass turbo
    def obtaining_public_transport(self):
        pass
            
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
        return _building.index[0]

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
        return index

    def cache_path(
        self,
        network : str,
        source_node: int,
        target_node: int,
        path: list[int],
    ) -> None:
        #print(f"caching path... current number of cached paths: {len(self._path_cache)}")
        cache = getattr(self, f"_cache_{network}")
        cache[(source_node, target_node)] = path      

    def get_cached_path(
        self, 
        network: str,
        source_node: int,
        target_node: int
    ) -> list[int]:
        cache = getattr(self, f"_cache_{network}")
        return cache.get((source_node, target_node), None)
    
    def _load_cache_files(self, layers : list[str]) -> None:
        for layer in layers:
            try:
                with open(f"/outputs/_cache/_cache_{layer}.pkl", "rb") as cache_file: 
                    setattr(self, f"_cache_{layer}", pickle.load(cache_file))
            except (FileNotFoundError, EOFError):
                setattr(self, f"_cache_{layer}", dict())
                warn(f"/outputs/_cache/_cache_{layer}.pkl not found, creating a new file", RuntimeWarning)

    def save_cache_files(self, layers : list[str]) -> None:
        for layer in layers:
            with open(f"/outputs/_cache/_cache_{layer}.pkl", "wb") as cache_file:
                pickle.dump(getattr(self, f"_cache_{layer}"), cache_file)
    
    def _weights_parser(self, agent: mesa.Agent, decision_rule : str):
        decision_rule = decision_rule.split(",")
        df = getattr(self, decision_rule[0]) 
        df = pd.eval(decision_rule[1].strip(), target=df)        
        return df.weights.replace(np.inf, 0)