import math #Extract working hour
from datetime import timedelta, datetime
import copy #Necessary for immutable objects in attributes dictionary

import pyproj #Type hinting for CRS
import numpy as np #Generate working hours
import warnings
import mesa
import mesa_geo as mg

import osmnx as ox
import networkx as nx

from shapely.geometry import Point, LineString
from shapely import distance

import scipy

import random
import geopandas as gpd
import pandas as pd

from streetcrime.utility import _proper_time_type

class Mover(mg.GeoAgent):
    """The Mover class, a subclass of GeoAgent, is the base class for all agents in the simulation. 
    It is able to move around the city using the shortest path in the road network.
    It has a unique id and a geometry (Point) that represents his position in the city. He also has a status (string) that describes what he is currently doing.
    At every step, the mover moves but only if he is travelling. If he's free, he gets a random activity to go to.
    The code is a modified version of project mesa/mesa-examples/gis/agents_and_networks/streetcrime/agent/commuter

    Parameters:
    ----------
        unique_id : int
            The unique id of the Mover.
        model : mesa.Model 
            The model of the simulation where the Mover is used. See streetcrime/model/model.py
        geometry : shapely.geometry.Point
            The point geometry of the Mover in the city.
        crs : pyproj.CRS 
            The crs of the Mover (usually same as mesa_geo.GeoSpace).
    
    Attributes:
    ----------
    - attributes : dict[str : str]
        It defines which attributes the Mover class has with respect to its parent class.
        It can be a value or a method. The method is used for the initialization of the attribute in Resident.data.
        It contains:
        - destination : dict[str, int | str | mesa.space.FloatCoordinate] : None
            - id : int
                -- The id of the building where the Mover is going to.
            - name : str
                -- The name of the building where the Mover is going to. (eg. home, work, day_act, night_act)
            - position : mesa.space.FloatCoordinate
                -- The position of the building where the Mover is going to.
        - step_in_path : int : None
            -- The current step in the path of the Mover.
        - path : list[mesa.space.FloatCoordinate] : None
            -- The current list of nodes that a travelling Mover has to follow to reach his destination.
    
    - params : dict[str, float]
        It contains fixed attributes or information on how the previously specified attributes are going to be generated.
        - walking_speed : float
            -- The speed of the Mover when walking (m/s). Default: 1.9
        - driving_speed : float
            -- The speed of the Mover when driving (m/s). Default: 8.3
        - max_walking_distance : float
            -- The maximum distance the Mover can walk (m) before driving. Default: 4000
        - mean_activity_length : float
            -- The mean length of an activity (hours). Default: 1
        - sd_activity_length : float
            -- The standard deviation of the length of an activity (hours). Default: 0.3        
    
    Methods:
    -------
    _gen_attribute(attribute_name : str, attribute_type : str = "float", distribution = "normal") -> float | datetime | list[int]
        -- Returns a random value based on type of attribute and distributuion
    initialize_attributes(attributes : list or str = None) -> None
        -- Initialize the `Mover.data[[attributes]]` with the values or methods contained in `Mover.attributes[[attributes]]`.
    step(): The step method of the Mover.
    go_to(building : str = None) -> None
        -- Sets specified building as `Mover.data['destination']['id']` as it finds it in the `Mover.data[f'{building}_id']`. 
        If it cannot be found, it gets a random building using `building` as function.
        
    __path_select() -> None
        -- It computes a path and changes his status to transport if the mover has a destination he is not already on the way.
    """
    walking_speed = 1.9, #m/s
    driving_speed = 7 #m/s
    car_use_threshold = 5000 #m
    sd_activity_end = 0.3
    mean_activity_end = 1
    car = True
    act_decision_rule = "buildings, weights = 1/df.geometry.distance(agent.geometry)"
    
    
    def __init__(self,
                 unique_id : int, 
                 model : mesa.Model, 
                 geometry : Point, 
                 crs : pyproj.CRS,
                 walking_speed : float = None,
                 driving_speed : float= None,
                 car_use_threshold : float = None,
                 sd_activity_end : float = None,
                 mean_activity_end : float = None,
                 car : bool = None,
                 act_decision_rule : str = None) -> None:
        """
        The __init__ method of the Mover class looks for the attributes and parameters of the subclasses, 
        adds them all together and assigns them to the current istance. It prioritize the model if it finds passed params and subclasses if it finds repeated attributes or params.
        
        Parameters:
        ----------
        unique_id : int
            The unique id of the Mover.
        model : mesa.Model 
            The model of the simulation where the Mover is used. See streetcrime/model/model.py
        geometry : shapely.geometry.Point
            The point geometry of the Mover in the city.
        crs : pyproj.CRS 
            The crs of the Mover (usually same as mesa_geo.GeoSpace).
        """
        super().__init__(unique_id, model, geometry, crs)
        if walking_speed is not None:
            self.walking_speed = walking_speed
        if driving_speed is not None:
            self.driving_speed = driving_speed
        if car_use_threshold is not None:
            self.car_use_threshold = car_use_threshold
        if sd_activity_end is not None:
            self.sd_activity_end = sd_activity_end
        if mean_activity_end is not None:
            self.mean_activity_end = mean_activity_end
        if car is not None:
            self.car = car
        if act_decision_rule is not None:
            self.act_decision_rule = act_decision_rule
        self.destination = {
            'id' : None,
            'name' : None,
            'node' : None
        }
        self.step_in_path = None
        self.path = None
        self.status = 'free'
        self.last_neighborhood = None
        self.activity_end_time = None
        self.network = None

    def step(self) -> None:
        """The Mover gets a random building if he is free, computes a path with an active destination if he hasn't started travelling, moves if he is travelling.
        """        
        if self.status == "busy":
            self._check_activity()
        if self.destination['id'] is None and self.status == "free":
            self.go_to('activity')
        if self.destination['id'] is not None and self.path is None:
            self._path_select()   
        if self.status == "transport":
            self._move()
    
    def go_to(self, 
              building : str = None) -> None:
        """Sets specified building as `Mover.data['destination']['id']` as it finds it in the `Mover.data[f'{building}_id']`. 
        If it cannot be found, it gets a random building using `building` as function.

        Parameters
        ----------
        building : str
            The building to go to. Can be "home", "work", "day_act", "night_act"

        Returns
        ----------
        None
        """
        if building == 'activity':
            if (self.model.datetime.replace(hour = self.model.day_act_end) >= 
                self.model.datetime >= 
                self.model.datetime.replace(hour = self.model.day_act_start)): 
                building = 'day_act'
            else:
                building = 'night_act'
        try:
            building_id = getattr(self, f'{building}')
        except AttributeError:
            building_id = self.model.space.get_random_building(function = building, 
                                                               agent = self,
                                                               decision_rule = self.act_decision_rule)
        self.destination['id'] = building_id
        self.destination['name'] = building
        self.destination['position'] = self.model.space.buildings.at[building_id, 'geometry'].centroid

    def _gen_attribute(self,
                       limits : list[float | datetime] = None,
                       attribute_type : str = "float", 
                       distribution = "normal",
                       mean : float = None,
                       sd : float = None,
                       a: float = None,
                       loc: float = None,
                       scale: float = None,
                       next_day : float = False) -> float | datetime | list[int]:
        """Returns a random value for the attribute `attribute_name` based on the specified type, and mean/sd/min/max specified in params.

        Parameters:
        ----------
        attribute_name : str
            the name of the attribute to generate 
        attribute_type : str, default="float"
            the type of the attribute to generate. Can be "float", "datetime_fixed", "datetime_variable"
        distribution : str, default="normal"
            the distribution used to generate. Can be "normal", "uniform"
            
        Returns:
        ----------
        attribute_value : float | datetime | list[int]
        """
        match distribution:
            case "normal":
                attribute_value = np.random.normal(mean, sd)
                if limits is not None:
                    while attribute_value < limits[0] or attribute_value > limits[1]:
                        attribute_value = np.random.normal(mean, sd)
                #Return based on the time
                if attribute_type in ["datetime_variable", "datetime_fixed", "timedelta"]:
                    attribute_value = _proper_time_type(time = attribute_value, 
                                                        attribute_type = attribute_type, 
                                                        len_step = self.model.len_step,
                                                        datetime = self.model.datetime,
                                                        next_day = next_day) 
            case "uniform":
                attribute_value = np.random.uniform(limits[0], limits[1])
            case "skewnorm":
                attribute_value = scipy.stats.skewnorm.rvs(a, loc, scale)
        return attribute_value
    
    def _check_activity(self) -> None:
        """Check if the agent has finished an activity"""
        if self.model.datetime >= self.activity_end_time:
            self.status = "free"
            self.activity_end_time = None
            
    def _path_select(self) -> None:
        """It computes a path and changes his status to transport if the mover has a destination he is not already on the way."""
       #Selects network
        i = 0
        path = self.__find_path()
        if self.destination['name'] == 'activity':
            while i<5 and path is None:
                i += 1
                self.go_to('activity')
                self.__find_path()
        
        if path is None:
            warnings.warn(f"Path cannot be found between agent current position and {self.destination['name']}. Assigning position of agent to destination.", RuntimeWarning)
            self.geometry = self.model.space.buildings.at[self.destination['id'], 'geometry'].centroid
            self.destination = {'id' : None, 'name' : None, 'node' : None}
            self.network = None
            return
        else:
            self.step_in_path = 0
            self.path = self.__redestribute_path_vertices(path)
            self.status = "transport"
            
    def _move(self) -> None:
        '''The method moves the mover across the path or, if he has arrived at the destination, it changes his status depending on the destination. 
        For activities, a random activity countdown is initiated.
        '''
        network_nodes = getattr(self.model.space, self.network + '_nodes')
        #Adds visits of Workers or PoliceAgents to the neighborhood
        today = str(self.model.datetime.date())
        column = None
        try:
            getattr(self, 'work_id')
            column = today + '_visits'
        except AttributeError:
            try:
                getattr(self, 'guardian')
                column = today + '_police'
            except AttributeError:
                pass
        #For Workers and PoliceAgents, add their visit to info_neighborhoods
        if column is not None:
            neighborhood_id = self.model.space.find_neighborhood_by_pos(self.geometry)
            if (self.last_neighborhood != neighborhood_id) & (neighborhood_id is not None):
                self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods[column].astype(float)
                self.model.info_neighborhoods.loc[(0, neighborhood_id), column] += 1
                self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods[column].astype(pd.SparseDtype(float, np.nan))
                self.last_neighborhood = neighborhood_id
        #If not arrived at destination
        if self.step_in_path < len(self.path):
            next_node = self.path[self.step_in_path]
            self.geometry = network_nodes.loc[next_node, 'geometry']
            self.step_in_path += 1
        #If arrived at destination
        elif self.step_in_path == len(self.path):
            self.geometry = self.model.space.buildings.at[self.destination['id'], 'geometry'].centroid
            self.step_in_path, self.path, self.network = None, None, None
            if self.destination['name'] in ['day_act', 'night_act']:
                self.activity_end_time = self._gen_attribute(limits = [0, float('inf')],
                                                                       attribute_type = "timedelta", 
                                                                       mean = self.mean_activity_end, 
                                                                       sd = self.sd_activity_end) 
                self.status = "busy"
            else:
                self.status = self.destination['name']
            self.destination = {'id' : None, 'name' : None, 'node' : None}

    def __find_path(self) -> list[mesa.space.FloatCoordinate]:
        if self.car and (distance(self.geometry, self.destination['position']) >= self.car_use_threshold):
            self.network = "roads"
        else:
            self.network = "public_transport"
        #TODO: the mover is always at a certain node so we could avoid this computation
        nearest_node = ox.distance.nearest_nodes(getattr(self.model.space, self.network), self.geometry.x, self.geometry.y)
        #TODO: should save node for every destination inside the dataframe to avoid double computation
        destination_node = ox.distance.nearest_nodes(getattr(self.model.space, self.network), self.destination['position'].x, self.destination['position'].y)
        
        path = self.model.space.get_cached_path(
            network = self.network,
            source_node = nearest_node,
            target_node = destination_node
        )
        
        if path is None:
            path = ox.distance.shortest_path(G = getattr(self.model.space, self.network), 
                                            orig = nearest_node,
                                            dest = destination_node,
                                            weight = 'travel_time',
                                            cpus=None)
            if path is not None:
                self.model.space.cache_path(
                    network = self.network,
                    source_node = nearest_node,
                    target_node = destination_node,
                    path=path,
                )       
        return path
        
    def __redestribute_path_vertices(self, original_path : list[mesa.space.FloatCoordinate]) -> list[mesa.space.FloatCoordinate]:
        """Returns the path but only with the nodes the mover will actually go through at each step, 
        depending on his speed and the length of the step.
        
        Parameters:
        ----------
        original_path : list[mesa.space.FloatCoordinate] 
            The original path to redistribute
        
        Returns:
        ----------
        redistributed_path : list[mesa.space.FloatCoordinate]
            The path the mover will go through at each step
        """
      
        network_nodes = getattr(self.model.space, self.network + '_nodes')
        network_edges = getattr(self.model.space, self.network + '_edges')
        
        # if origin and destination are the same, no need to redistribute path vertices.
        if len(original_path) > 1:
            nodes_in_path = network_nodes.loc[original_path, :]
            nodes_in_path.reset_index(inplace=True)
            nodes_in_path.loc[:, 'previous_node'] = nodes_in_path.loc[:, 'osmid'].shift(1)
            edges_in_path = list(zip(nodes_in_path[1:]['previous_node'], nodes_in_path[1:]['osmid'], [0]*len(nodes_in_path[1:])))
            
            def get_edges_rows(edge):
                try:
                    return network_edges.xs(edge, level=['u', 'v', 'key']).iloc[0]
                except (IndexError, KeyError):
                    try:
                        return network_edges.xs(edge, level=['v', 'u', 'key']).iloc[0]
                    except (IndexError, KeyError):
                        warnings.warn(f"Path cannot be found between agent current position and {self.destination['name']}. Assigning position of agent to destination.", RuntimeWarning)
                        self.geometry = self.model.space.buildings.at[self.destination['id'], 'geometry'].centroid
                        self.destination = {'id' : None, 'name' : None, 'node' : None}
                        self.network = None
                        return None
            edges_in_path = [get_edges_rows(edge) for edge in edges_in_path]
            if any(isinstance(edge, type(None)) for edge in edges_in_path):
                return
            edges_in_path = gpd.GeoDataFrame(edges_in_path)
            edges_in_path.loc[:, 'cumulative_travel_time'] = edges_in_path['travel_time'].cumsum()
            edges_in_path.loc[:, 'cumulative_travel_time'] = edges_in_path['cumulative_travel_time']/(self.model.len_step*60)
            total_steps = edges_in_path['cumulative_travel_time'].iloc[-1]
            redistributed_path = []
            for i in range(1, math.ceil(total_steps)+1):
                try:
                    redistributed_path.append(edges_in_path.loc[edges_in_path['cumulative_travel_time'] >= i, :].index[0][0])
                except IndexError:
                    redistributed_path.append(nodes_in_path['osmid'].iloc[-1])
        else:
            redistributed_path = original_path
        return redistributed_path 