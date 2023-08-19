from typing import List, Any #Type hinting
import math #Extract working hour

import pyproj #Type hinting for CRS

from types import MethodType #Type hinting for methods
import numpy as np #Generate working hours

from datetime import timedelta, datetime
import geopandas as gpd #Type hinting for geodataframe

import mesa
import mesa_geo as mg
import shapely
from shapely.geometry import Point, LineString

from src.space.utils import redistribute_vertices, UnitTransformer
from src.space.road_network import RoadNetwork

class Mover(mg.GeoAgent):
    """The Mover class, a subclass of GeoAgent, is the base class for all agents in the simulation. 
    He is able to move around the city using the shortest path in the road network.
    He has a unique id and a geometry (Point) that represents his position in the city. He also has a status (string) that describes what he is currently doing.
    At every step, the mover moves but only if he is travelling. The code is a modified version of project mesa/mesa-examples/gis/agents_and_networks/src/agent/commuter

    Arguments:
        unique_id (int) -- The unique id of the Mover.
        model (mesa.Model) -- The model of the simulation where the Mover is used. See src\model\model.py
        geometry (shapely.geometry.Point) -- The point geometry of the Mover in the city.
        crs (pyproj.CRS) -- The crs of the Mover (usually same as mesa_geo.GeoSpace).
    
    attribute_names:
        params (dict[str, float]) -- The parameters of the Mover chosen when creating the model. See src\model\model.py
        _attributes (dict[str, mesa.space.FloatCoordinate or int or List(mesa.space.FloatCoordinate)])
            -- The attributes of the Mover. It contains:
                status: str. Worker -- current status of the Mover (eg. "home", "work", "busy", "free", "transport")
                path (List[mesa.space.FloatCoordinate]) -- The current list of nodes that a travelling Mover has to follow to reach his destination.
                origin (mesa.space.FloatCoordinate) -- The original position of the Mover's current trip.
                destination (mesa.space.FloatCoordinate) -- The destination of the Mover's current trip.
                step_in_path (int) -- The current step in the path of the Mover.         
    """
    attributes  : dict[str, mesa.space.FloatCoordinate or int or List(mesa.space.FloatCoordinate)] = {
        'origin' : None,
        'destination' :  None,
        'step_in_path' : None,
        'path' : None
        }
    params: dict[str, float] = {
        "walking_speed": 1.9, #m/s
        "driving_speed": 8.3, #m/s
        "max_walking_distance" : 4000, #m
        "mean_activity_length" : 1,
        "sd_activity_length" : 0.3
        }
    
    def __init__(self, unique_id : int, model : mesa.Model, geometry : shapely.geometry.Point, crs : pyproj.CRS) -> None:
        """
        The __init__ method of the Mover class looks for the attributes and parameters of the subclasses, 
        adds them all together and assigns them to the current istance.
        
        Arguments:
            unique_id (int) -- The unique id of the Mover.
            model (mesa.Model) -- The model of the simulation where the Mover is used. See src\model\model.py
            geometry (shapely.geometry.Point): The point geometry of the Mover in the city.
            crs (pyproj.CRS): The crs of the Mover (usually same as mg.GeoSpace).
        """
        super().__init__(unique_id, model, geometry, crs)
        self.data = {}
        parent_classes = self.__class__.__mro__[1:-4]
        #Unite every attribute of parent classe
        for class_obj in parent_classes:
            self.attributes.update(class_obj.attributes)
            self.params.update(class_obj.params)
        self.initialize_attributes()
        
    def initialize_attributes(self, attributes : list = None) -> None:
        if attributes is None:
            attributes = self.attributes.keys()
        for attribute in attributes:
            if isinstance(attribute, MethodType):
                self.data[attribute] = attribute()
            else:
                self.data[attribute] = attribute

    def step(self) -> None:
        """The Mover gets a random building if he is free, computes a path with an active destination if he hasn't started travelling, moves if he is travelling."""
        if self.data['status'] == "busy":
            self._check_activity()
        if self.data['destination'] is None and self.data['status'] == "free":
            #TODO: Convert opening times to parameters
            if (self.data['datetime'] >= self.data['datetime'].replace (hour = 8)) and (self.data['datetime'] <= self.data['datetime'].replace(hour = 19)):
                self.go_to("day_act")
        else:
            self.go_to("night_act")
        self._prepare_to_move()   
        self._move()
    
    def _prepare_to_move(self) -> None:
        if (self.data['step_in_path'] is None) and (self.data['destination'] is not None):
            self.data['path'] = self._path_select()
            self.data['status'] = "transport"
            
    def _path_select(self) -> List[mesa.space.FloatCoordinate]:
        """It tries to find the cached path in "outputs\_path_cache_result.pkl", otherwise computes it. 
        Finally it finds the nodes for each step according to the mover_speed specified in params 

        Returns:
            path (List[mesa.space.FloatCoordinate]) -- List of nodes the mover will have to go through at each step
        """
        self.data['step_in_path'] = 0
        if (
            cached_path := self.model.space.roads.get_cached_path(
                source=(self.geometry.centroid.x, self.geometry.centroid.y),
                target=self.data['destination']
            )
        ) is not None:
            path = cached_path
        else:
            path = self.model.space.roads.get_shortest_path(
                source=(self.geometry.centroid.x, self.geometry.centroid.y), 
                target=self.data['destination']
            )
            self.model.space.roads.cache_path(
                source=(self.geometry.centroid.x, self.geometry.centroid.y),
                target=self.data['destination'],
                path=path,
            )
        path = self._redistribute_path_vertices(path)
        return path
    
    def _redistribute_path_vertices(self, original_path : List[mesa.space.FloatCoordinate]) -> None:
        """Returns the path but only with the nodes the mover will actually go through at each step.
        
        Arguments:
            original_path(List[mesa.space.FloatCoordinate]) -- the original path to redistribute
        Returns:
            (List[mesa.space.FloatCoordinate]) -- the path the mover will follow
        """
        # if origin and destination are the same, no need to redistribute path vertices.
        if len(original_path) > 1:
            original_path = LineString([Point(p) for p in original_path])
            if original_path.length < self.params['max_walking_distance']:
                speed = self.params['walking_speed']
            else:
                speed = self.params['driving_speed']
            redistributed_path_in_meters = redistribute_vertices(
                original_path, speed*self.model.model_params['len_step']*60 #From m/s to m/step
            )
            return list(redistributed_path_in_meters.coords)
                
    def _move(self) -> None:
        '''The method moves the mover across the path or, if he has arrived at the destination, it changes his status depending on the destination. 
        For activities, a random activity countdown is initiated.
        '''
        if self.data['status'] == "transport":
            if self.data['step_in_path'] < len(self.data['path']):
                next_position = self.data['path'][self.data['step_in_path']]
                self.geometry = Point(next_position)
                self.data['step_in_path'] += 1
                if self.data.has_key('work_id'):
                    _neighborhood_id = self.model.space.find_neighborhood_by_position(self.geometry)
                    if last_neighborhood != _neighborhood_id:
                        self.model.space.neighborhoods_df.at[last_neighborhood, 'today_visits'] += 1
                        last_neighborhood = _neighborhood_id
            elif self.data['step_in_path'] == len(self.data['path']):
                self.data['step_in_path'], self.data['path'] = None
                if self.data.has_key('work_id'):
                    if self.data['destination'] == self.model.space.buildings_df.at[self.data['work_id'], 'entrance']:
                        self.data['status'] = "work"
                        return
                if self.data.has_key('home_id'):
                    if self.data['destination'] == self.model.space.buildings_df.at[self.data['home_id'], 'entrance']:
                        self.data['status'] = "home"
                        return
                self.data['activity_end_time'] = self._gen_time('activity_end')
                self.data['status'] = "busy"

    def go_to(self, building_type : str) -> None:
        try:
            _building_id = self.data[f'{building_type}_id']
        except KeyError:
            _building_id = self.model.space.get_random_building(self, function = building_type)
        self.data['destination'] = self.model.space.buildings_df.at[_building_id, 'entrance']

    def _check_activity(self) -> None:
        """Check if the agent has finished his activity"""
        if self.model.data['datetime'] >= self.data['activity_end_time']:
            self.data['status'] = "free"
            self.data['activity_end_time'] = None
                      
    def _get_yesterday_info(self, info_type : str = None) -> gpd.GeoDataFrame:
        """Gets a random sample of p_information from the model data of the previous day, groups it by neighborhood 
        and returns the dataset.
    
        Arguments:
            info_type (str) -- can be "crimes"
            
        Returns:
            (gpd.GeodataFrame) -- the dataset with the information of the previous day
            
        """
        known_data = self.model.model_data[f'{info_type}'].sample(frac = self.params['p_information'])
        _data_per_neighborhood = known_data['neighborhood'].value_counts()
        for neighborhood, data in _data_per_neighborhood.items():
            self.data[f'{info_type}_neighborhoods'].at[neighborhood, f'yesterday_{data}'] = data
    
    def gen_attribute(self, attribute_name : str, attribute_type : str = "float", distribution = "normal") -> float:
        """Changes the self.data[f'{attribute_name}'] to a random value for the next activity based on 
        the mean, the sd and the limits specified in params

        Arguments:
            attribute_name -- can be "self_defence", "crime_motivation"
        """
        match attribute_type:
            case "float":
                limits = [0,1]
            case "datetime_fixed", "datetime_variable":
                limits = [float('-inf'),float('+inf')]
                def __next_day(time : float) -> int:
                    if time >= 24 or attribute_type == "resting_end":
                        return 1
                    else:
                        return 0
                def __adjust_time(time : float) -> datetime:
                    rounded_minutes = self.model.model_params['len_step'] * round(float_time * 60 / self.model.model_params['len_step'])
                    if attribute_type == "datetime_fixed":
                        adj_time = [math.floor(rounded_minutes/60), rounded_minutes%60]
                    else:
                        _minutes = timedelta(minutes = rounded_minutes)
                        adj_time = self.model.data['datetime'].replace(hour = _minutes.seconds//3600, minute = (_minutes.seconds//60)%60) + timedelta(days = next_day)
        mean = self.params[f'mean_{attribute_name}']
        sd = self.params[f'sd_{attribute_name}']
        try: 
            limits[0] = self.params[f'min_{attribute_name}']
        except KeyError:
            pass
        try: 
            limits[1] = self.params[f'max_{attribute_name}']
        except KeyError:
            pass
        attribute_value = np.random.normal(mean, sd)
        if attribute_type == "datetime":
            next_day = __next_day(attribute_value)
        while attribute_value < limits[0] or attribute_value > limits[1]:
            attribute_value = np.random.normal(mean, sd)
            if attribute_type == "datetime":
                next_day = __next_day(attribute_value)
        if attribute_type == "datetime":
            return __adjust_time(attribute_value)
        else:
            return attribute_value