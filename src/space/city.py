from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import mesa 
import mesa_geo as mg
import shapely
from shapely.geometry import Point
import os.path
from src.agents.mover import Mover
from src.agents.resident import Resident
from src.agents.worker import Worker
from src.agents.criminal import Criminal
from networkx import MultiDiGraph
import osmnx as ox
import pickle
import random
import ast
import warnings

    
class City(mg.GeoSpace):
    """
    The City class is the GeoSpace in which agents move.
    It is used to store the road & public transport network, the buildings and neighborhoods dataframes.
    
    Parameters:
    ----------
        crs : pyproj.CRS
            The crs of the city
        roads : gpd.GeoDataFrame, optional)
        
    Methods:
        get_random_building(resident : Resident, function : str) -> int
            Returns a random building id based on the function passed as argument
        find_neighborhood_by_pos(position : Point) -> int
            Returns the neighborhood id of the neighborhood in which the position passed as argument is contained
        distance_from_buildings(position : Point) -> gpd.GeoSeries
            Returns the distance from the position passed as argument to all the buildings in the city
        update_information() -> None
            Updates 'yesterday_crimes' and 'yesterday_visits' columns of the buildings dataframe with data
            collected in 'today_crimes' and 'today_visits' columns of the neighborhoods dataframe
    """
    
    def __init__(self, 
                crs: str,
                directory: str = None,
                model : mesa.Model = None,
                roads_nodes : gpd.GeoDataFrame = None,
                roads_edges : gpd.GeoDataFrame = None,
                roads: MultiDiGraph = None,
                public_transport_nodes : gpd.GeoDataFrame = None,
                public_transport_edges : gpd.GeoDataFrame = None,
                public_transport: MultiDiGraph = None,
                neighborhoods:gpd.GeoDataFrame = None, 
                buildings:gpd.GeoDataFrame = None,
                **other_layers: gpd.GeoDataFrame | mg.RasterLayer | mg.ImageLayer) -> None:
        #TODO: specify columns of the dataframes
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
        super().__init__(crs=crs)
        self.model = model
        self.directory = directory
        layers = {'roads' : roads,
                  'roads_nodes' : roads_nodes,
                  'roads_edges' : roads_edges,
                  'public_transport_nodes' : public_transport_nodes,
                  'public_transport_edges' : public_transport_edges,
                  'public_transport' : public_transport,
                  'neighborhoods' : neighborhoods, 
                  'buildings' : buildings}
        layers.update(**other_layers)
        self.__dict__.update(layers)
        #This is purely for visualization purposes.
        #for layer in layers.values():
            #if not isinstance(layer, MultiDiGraph):
            #    self.add_layer(layer)
        self._load_cache_files(['roads', 'public_transport'])

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
                def _sample_home(weights : gpd.GeoSeries) -> gpd.GeoDataFrame:
                    _neighborhood = self.neighborhoods.sample(n=1, weights=weights)
                    return self.buildings[(self.buildings['home'] == True) & (self.buildings['neighborhood'] == _neighborhood.index[0])].sample(n = 1)
                match type(resident).__qualname__:
                    case Criminal.__qualname__:
                        #Weights = proportion of population in each neighborhood * (1/income)
                        weights = self.neighborhoods['prop'] * (1/self.neighborhoods['mean income'])
                        _building = _sample_home(weights)
                    case Worker.__qualname__:
                        #Weights = proportion of population in each neighborhood 
                        weights = self.neighborhoods['prop']
                        #Check that it's different from the work building
                        _building = _sample_home(weights)
                        while _building.index[0] == resident.data['work_id']:
                            _building = _sample_home(weights)
                    case _:
                        #Weights = proportion of population in each neighborhood
                        weights = self.neighborhoods['prop']
                        _building = _sample_home(weights)    
            case "day_act" | "night_act":
                    #On the first day, residents don't have information thus get a building based on distance
                    if resident.model.data['datetime'].day == 1:
                        weights = 1/self.distance_from_buildings(resident.geometry)
                    else:
                        columns_names = [
                                node.id for node in ast.walk(ast.parse(resident.params['act_decision_rule'])) 
                                if isinstance(node, ast.Name)
                                ]
                        if 'distance' in columns_names:
                            distance = self.distance_from_buildings(resident.geometry)
                            columns_names.remove('distance')
                        if 'mean_income' in columns_names:
                            mean_income = self.buildings['mean_income']
                            columns_names.remove('mean_income')
                        if resident.params['p_information'] == 1:
                            columns = [self.buildings[column_name] for column_name in columns_names]
                        else:
                            _buildings = self.buildings.copy(deep=True)
                            _buildings.drop(columns_names, axis='columns', inplace = True)
                            relevant_info = resident.model.data['info_neighborhoods'].loc[resident.unique_id, columns_names]
                            _buildings = _buildings.merge(relevant_info, left_on = 'neighborhood', right_index = True)
                            columns = [_buildings[column_name] for column_name in columns_names] 
                        for name, column in zip(columns_names, columns):
                            exec(f"{name} = column")                            
                        weights = eval(resident.params['act_decision_rule'])
                    weights = weights.astype(float)
                    weights.replace(np.inf, 0, inplace=True)
                    _building = self.buildings[(self.buildings[function] == True)].sample(n=1, weights=weights)                            
            case _:
                _building = self.buildings.sample(n=1)
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
            warnings.warn(f"Position {position} is not contained in any neighborhood", RuntimeWarning)
            index = None
        return index
            
    def distance_from_buildings(self, position: Point) -> gpd.GeoSeries:
        """Find the distance from the position passed as argument to all the buildings in the city

        Args:
            position (Point): The position to find the distance from

        Returns:
            gpd.GeoSeries: The distances from the position to all the buildings in the city, ordered as self.buildings
        """
        return self.buildings['geometry'].distance(position)

    def cache_path(
        self,
        network_name : str,
        source_node: mesa.space.FloatCoordinate,
        target_node: mesa.space.FloatCoordinate,
        path: list[mesa.space.FloatCoordinate],
    ) -> None:
        #print(f"caching path... current number of cached paths: {len(self._path_cache)}")
        cache = getattr(self, f"_cache_{network_name}")
        cache[(source_node, target_node)] = path
        

    def get_cached_path(
        self, 
        network_name: str,
        source_node: mesa.space.FloatCoordinate,
        target_node: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate]:
        cache = getattr(self, f"_cache_{network_name}")
        return cache.get((source_node, target_node), None)
    
    def _load_cache_files(self, layers : list[str]) -> None:
        for layer in layers:
            setattr(self, f"_path_cache_{layer}", self.directory + f"\outputs\_cache_{layer}.pkl")
            try:
                with open(getattr(self, f"_path_cache_{layer}"), "rb") as cached_result: 
                    setattr(self, f"_cache_{layer}", pickle.load(cached_result))
            except (FileNotFoundError, EOFError):
                setattr(self, f"_cache_{layer}", dict())

    def save_cache_files(self, layers : list[str]) -> None:
        for layer in layers:
            with open(getattr(self, f"_path_cache_{layer}"), "wb") as cache_file:
                pickle.dump(getattr(self, f"_cache_{layer}"), cache_file)