import time
import geopandas as gpd
import numpy as np
import mesa 
import mesa_geo as mg
from shapely.geometry import Point
import os.path
from src.agent.mover import Mover
from src.agent.resident import Resident
from src.agent.worker import Worker
from src.agent.criminal import Criminal
from networkx import MultiDiGraph
import osmnx as ox
import pickle
import random

current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]
    current_directory = parent_directory
    
class City(mg.GeoSpace):
    """
    The City class is the GeoSpace in which agents move.
    It is used to store the road network and the buildings and neighborhoods dataframes.
    ...
    
    Arguments:
        crs(str)
            The crs of the city
        road_network(gpd.GeoDataFrame, optional)
            The road network of the city. Will be converted to a RoadNetwork object (see src/space/road_network)
        neighborhoods(gpd.GeoDataFrame, optional)
            The neighborhoods dataframe of the city
        buildings_df(gpd.GeoDataFrame, optional)
            The buildings dataframe of the city
    Methods:
        get_random_building(resident : Resident, function : str) -> int
            Returns a random building id based on the function passed as argument
        find_neighborhood_by_position(position : Point) -> int
            Returns the neighborhood id of the neighborhood in which the position passed as argument is contained
        distance_from_buildings(position : Point) -> gpd.GeoSeries
            Returns the distance from the position passed as argument to all the buildings in the city
        update_information() -> None
            Updates 'yesterday_crimes' and 'yesterday_visits' columns of the buildings dataframe with data
            collected in 'today_crimes' and 'today_visits' columns of the neighborhoods dataframe
    """
    
    def __init__(self, 
                 crs: str, 
                 road_network: MultiDiGraph = None, 
                 neighborhoods:gpd.GeoDataFrame = None, 
                 buildings:gpd.GeoDataFrame = None,
                **other_layers: gpd.GeoDataFrame | mg.RasterLayer | mg.ImageLayer) -> None:
        #TODO: specify columns of the dataframes
        """
            Arguments:
            crs(str)
                The crs of the city
            road_network(networkx.MultiDiGraph, optional)
                The road network of the city. Will be converted to a RoadNetwork object (see src/space/road_network)
            neighborhoods(gpd.GeoDataFrame, optional)
                The neighborhoods dataframe of the city
            buildings_df(gpd.GeoDataFrame, optional)
                The buildings dataframe of the city
        """
        super().__init__(crs=crs)
        layers = {'road_network' : road_network,
                  'roads_nodes' : ox.graph_to_gdfs(road_network, nodes=True, edges=False),
                  'neighborhoods' : neighborhoods, 
                  'buildings' : buildings}
        layers.update(**other_layers)
        self.__dict__.update(layers)
        #This is purely for visualization purposes.
        #for layer in layers.values():
            #if not isinstance(layer, MultiDiGraph):
            #    self.add_layer(layer)
        self._path_cache_result = os.path.join(current_directory, "outputs\_path_cache_result.pkl")
        try:
            with open(self._path_cache_result, "rb") as cached_result: #"rb" = read binary", "with" allows to open and close after execution 
                self._path_select_cache = pickle.load(cached_result)
        except (FileNotFoundError, EOFError):
            self._path_select_cache = dict()
        #TEST

    def get_random_building(self, 
                            function: str = None,
                            resident : Resident = None)-> int:
        """This method act on the City.buildings_df to obtain a random building id based on the function and the Resident that requested it.
        Args:
            resident (Resident, optional): The resident that is looking for a building, by default None.
            function (str, optional): The function of the building that the resident is looking for, by default None. Can be "home", "work", "day_act" or "night_act".  
        
        Returns:
            int: The id of the building
        
        """
        match function:
            case "home":
                match type(resident).__qualname__:
                    case Criminal.__qualname__:
                        #Weights = proportion of population in each neighborhood * (1/income)
                        weights = self.neighborhoods['prop'] * (1/self.neighborhoods['income'])
                    case _:
                        #Weights = proportion of population in each neighborhood
                        weights = self.neighborhoods['prop']
                _neighborhood_df = self.neighborhoods.sample(n=1, weights=weights) #TODO: TEST
                #_building = self.buildings[(self.buildings['home'] == True) & (self.buildings['neighborhood'] == _neighborhood_df.index[0])].sample(n = 1)
                # TODO: RIGHT ONE BUT USE IT ONLY WITH ALL BUILDINGS, use only building df at the end of the function to avoid repetition
                _building = self.buildings[self.buildings['home'] == True].sample(n=1)
                if type(resident).__qualname__ == Worker.__qualname__:
                    while _building.index[0] == resident.data['work_id']:
                        _building = self.buildings[self.buildings['home'] == True].sample(n=1)
            case "day_act" | "night_act":
                    if resident.model.data['datetime'].day == 1:
                        #Weights = distance from the resident
                        weights = 1/self.distance_from_buildings(resident.geometry)
                    else:
                        match type(resident).__qualname__:
                            case Worker.__qualname__:
                                #The worker chooses based on the distance from him and the known number of yesterday crimes in the neighborhood
                                #TODO: Instead of creating a deep copy, merge columns to existing self.buildings by renaming them like 'yesterday_crimes_resident'? 
                                _buildings = self.buildings.copy(deep=True)
                                _buildings.drop(['yesterday_crimes', 'run_crimes'], axis='columns', inplace = True)
                                _buildings = _buildings.merge(resident.data['info_neighborhoods'], left_on = 'neighborhood', right_index = True)
                                weights = (1/self.distance_from_buildings(resident.geometry)) * (1/_buildings['yesterday_crimes']) * (1/_buildings['run_crimes'])
                            case Criminal.__qualname__:
                                #The criminal chooses based on the distance from him and the known number of yesterday visits in the neighborhood
                                weights = (1/self.distance_from_buildings(resident.geometry)) * (1/self.buildings['yesterday_visits'])
                            case _:
                                weights = 1/self.distance_from_buildings(resident.geometry)
                    weights.replace(np.inf, 0, inplace=True)
                    _building = self.buildings[(self.buildings[function] == True)].sample(n=1, weights=weights)
                    #Avoiding that the same building gets selected
                    while _building['entrance_node'].iloc[0] == ox.nearest_nodes(self.road_network, 
                                                                         resident.geometry.x,
                                                                         resident.geometry.y):
                        _building = self.buildings[(self.buildings[function] == True)].sample(n=1, weights=weights)
            case _:
                _building = self.buildings.sample(n=1)
        return _building.index[0]

    def find_neighborhood_by_position(self, position: Point) -> int:
        """Find the neighborhood in which the position passed as argument is contained

        Args:
            position (Point): The position to find the containing neighborhood

        Returns:
            int: The id of the neighborhood
        """
        return self.neighborhoods[self.neighborhoods['geometry'].contains(position)].index[0]

    def distance_from_buildings(self, position: Point) -> gpd.GeoSeries:
        """Find the distance from the position passed as argument to all the buildings in the city

        Args:
            position (Point): The position to find the distance from

        Returns:
            gpd.GeoSeries: The distances from the position to all the buildings in the city, ordered as self.buildings
        """
        return self.buildings['geometry'].distance(position)

    def update_information(self) -> None:
        """Updates 'yesterday_crimes' and 'yesterday_visits' columns of the self.buildings with data
        collected in 'today_crimes' and 'today_visits' columns of the self.neighborhoods dataframe.
        This method is only initiated if the model.worker_params.information = 1 (Perfect Information).
        """
        self.neighborhoods['run_visits'] += self.neighborhoods['yesterday_visits'] - 1
        self.neighborhoods['run_crimes'] += self.neighborhoods['yesterday_crimes'] - 1
        self.neighborhoods['run_police'] += self.neighborhoods['yesterday_police'] - 1
        self.buildings.drop(['yesterday_visits', 
                             'yesterday_crimes',
                             'yesterday_police',
                             'run_visits', 
                             'run_crimes',
                             'run_police'], inplace=True, axis='columns')
        _info_neighborhoods = self.neighborhoods.copy(deep = True)
        _info_neighborhoods = _info_neighborhoods[['yesterday_visits', 'yesterday_crimes', 'yesterday_police', 'run_visits', 'run_crimes', 'run_police']]
        self.buildings = self.buildings.merge(
            _info_neighborhoods, left_on='neighborhood', right_index = True)
        # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
        self.neighborhoods = self.neighborhoods.assign(
            yesterday_visits=1, yesterday_crimes=1, yesterday_police =1)

    def cache_path(
        self,
        source_node: mesa.space.FloatCoordinate,
        target_node: mesa.space.FloatCoordinate,
        path: list[mesa.space.FloatCoordinate],
    ) -> None:
        #print(f"caching path... current number of cached paths: {len(self._path_select_cache)}")
        self._path_select_cache[(source_node, target_node)] = path
        self._path_select_cache[(target_node, source_node)] = list(reversed(path))
        with open(self._path_cache_result, "wb") as cached_result:
            pickle.dump(self._path_select_cache, cached_result)

    def get_cached_path(
        self, source_node: mesa.space.FloatCoordinate, target_node: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate]:
        return self._path_select_cache.get((source_node, target_node), None)