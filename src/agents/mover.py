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

from src.space.utils import redistribute_vertices

import scipy

import random
import geopandas as gpd
import pandas as pd

class Mover(mg.GeoAgent):
    """The Mover class, a subclass of GeoAgent, is the base class for all agents in the simulation. 
    It is able to move around the city using the shortest path in the road network.
    It has a unique id and a geometry (Point) that represents his position in the city. He also has a status (string) that describes what he is currently doing.
    At every step, the mover moves but only if he is travelling. If he's free, he gets a random activity to go to.
    The code is a modified version of project mesa/mesa-examples/gis/agents_and_networks/src/agent/commuter

    Parameters:
    ----------
        unique_id : int
            The unique id of the Mover.
        model : mesa.Model 
            The model of the simulation where the Mover is used. See src/model/model.py
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
        
    _prepare_to_move() -> None
        -- It computes a path and changes his status to transport if the mover has a destination he is not already on the way.
    """
    attributes  : dict[str, dict[str, int or str or mesa.space.FloatCoordinate] or int or list(mesa.space.FloatCoordinate)] = {
        'destination' : {
            'id' : None,
            'name' : None,
            'node' : None
        },
        'step_in_path' : None,
        'path' : None,
        'status' : None,
        'last_neighborhood' : None,
        'car' : False
        }
    
    params: dict[str, float] = {
        "walking_speed": 1.9, #m/s
        "driving_speed": 7, #m/s
        "mean_activity_end" : 1,
        "sd_activity_end" : 0.3,
        "car_use_threshold" : 5000, #m
        }
    
    def __init__(self, 
                 unique_id : int, 
                 model : mesa.Model, 
                 geometry : Point, 
                 crs : pyproj.CRS) -> None:
        """
        The __init__ method of the Mover class looks for the attributes and parameters of the subclasses, 
        adds them all together (prioritizing the subclass if it finds repeated attributes or params) and assigns them to the current istance.
        
        Parameters:
        ----------
        unique_id : int
            The unique id of the Mover.
        model : mesa.Model 
            The model of the simulation where the Mover is used. See src/model/model.py
        geometry : shapely.geometry.Point
            The point geometry of the Mover in the city.
        crs : pyproj.CRS 
            The crs of the Mover (usually same as mesa_geo.GeoSpace).
        """
        self.data = {'unique_id' : unique_id}
        super().__init__(unique_id, model, geometry, crs)
        classes = self.__class__.__mro__[:-4]
        #Unite every attribute of parent classes
        for class_obj in classes:
            for attribute_name, attribute_value in class_obj.attributes.items():
                self.attributes.setdefault(attribute_name, attribute_value)
        #Unite every parameter of parent classes, giving precedence to the ones specified in the model
            for params_name, params_value in class_obj.params.items():
                if self.model.agents_params is not None:
                    class_params = self.model.agents_params.get(class_obj.__name__, {})
                    self.params.setdefault(params_name, class_params.get(params_name, params_value))
        self._initialize_attributes()
    
    @property
    def geometry(self):
        return self._geometry
    
    @geometry.setter
    def geometry(self, new_geometry):
        self._geometry = new_geometry
        self.data['geometry'] = new_geometry

    def step(self) -> None:
        """The Mover gets a random building if he is free, computes a path with an active destination if he hasn't started travelling, moves if he is travelling.
        """        
        if self.data['status'] == "busy":
            self._check_activity()
        if self.data['destination']['id'] is None and self.data['status'] == "free":
            if (self.model.data['datetime'] >= 
                self.model.data['datetime'].replace(hour = self.model.model_params['day_act_start']) 
                and self.model.data['datetime'] <= 
                self.model.data['datetime'].replace(hour = self.model.model_params['day_act_end'])):
                self.go_to("day_act")
            else:
                self.go_to("night_act")
        self._prepare_to_move()   
        self._move()
    
    def go_to(self, building : str = None) -> None:
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
        try:
            building_id = self.data[f'{building}_id']
        except KeyError:
            building_id = self.model.space.get_random_building(function = building, resident = self)
        self.data['destination']['id'] = building_id
        self.data['destination']['name'] = building
        self.data['destination']['position'] = self.model.space.buildings.at[building_id, 'geometry'].centroid

    def _initialize_attributes(self, attributes : list or str = None) -> None:
        """Initialize the `Mover.data[[attributes]]` with the values or methods contained in `Mover.attributes[[attributes]]`.
        If `attributes` is None, it initializes all attributes.

        Parameters:
        ----------
        attributes : list or str, default=None
            The list of attributes to initialize. If None, all attributes are initialized.
        
        Returns:
        ----------
        None
        """
        if attributes is None:
            attributes = self.attributes
        else:
            attributes = {attribute_name : self.attributes[attribute_name] for attribute_name in attributes}
        for attribute_name, attribute_value in attributes.items():
            try :
                self.data[attribute_name] = eval("self."+attribute_value)
            except TypeError:
                self.data[attribute_name] = copy.deepcopy(attribute_value)

    def _gen_attribute(self, 
                      attribute_name : str, 
                      attribute_type : str = "float", 
                      distribution = "normal") -> float | datetime | list[int]:
    #TODO: Update the distribution implementation
    #TODO: Maybe brings next_day & adjust time to utils
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
        if attribute_type in ["datetime_fixed", "datetime_variable"]:
            limits = [float('-inf'), float('+inf')]
            if attribute_name in ["resting_start_time", "resting_end_time"]:
                if self.model.data['datetime'] == self.model.model_params['start_datetime']: 
                    if attribute_name == "resting_start_time":
                        return self.model.model_params['start_datetime']
                    elif attribute_name == "resting_end_time":
                        return self.model.model_params['start_datetime'] + timedelta(hours = np.random.normal(1, 0.5))
                elif attribute_name == "resting_start_time":
                    try:
                        limits[0] = self.data['work_end_time'][0] + self.data['work_end_time'][1]/60
                    except:
                        pass
                elif attribute_name == "resting_end_time":
                    try:
                        limits[1] = self.data['work_start_time'][0] + self.data['work_start_time'][1]/60
                    except:
                        pass
        elif attribute_type == "float":
            if attribute_name == "income":
                limits = [0, float('+inf')]
                neighborhood = self.model.space.buildings.at[self.data['home_id'], 'neighborhood']
                ae = self.model.space.neighborhoods.at[neighborhood, 'ae']
                loce = self.model.space.neighborhoods.at[neighborhood, 'loce']
                scalee = self.model.space.neighborhoods.at[neighborhood, 'scalee']
                attribute_value = scipy.stats.skewnorm.rvs(a = ae, loc = loce, scale = scalee)
                while attribute_value < limits[0] or attribute_value > limits[1]:
                    attribute_value = scipy.stats.skewnorm.rvs(a = ae, loc = loce, scale = scalee)
                return attribute_value
            elif attribute_name == "crime_motivation":
                attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.data['income']) + np.random.normal(0, 0.10)
                while attribute_value < 0 or attribute_value > 1:
                    attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.data['income']) + np.random.normal(0, 0.10)
                return attribute_value
            elif attribute_name == "crime_attractiveness":
                attribute_value = self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.data['income']) + np.random.normal(0, 0.10)
                while attribute_value < 0 or attribute_value > 1:
                    attribute_value = self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.data['income']) + np.random.normal(0, 0.10)
                return attribute_value
            else:
                limits = [0,1] 
        elif attribute_type == "bool":
            if attribute_name == "car":
                prob = self.data['income']/self.params['car_income_threshold'] + np.random.normal(0, 0.1)
                if prob < 0.5:
                    return False
                else:
                    return True 
        elif attribute_type == "timedelta":
            limits = [0, float('+inf')]
        try: 
            limits[0] = self.params[f'min_{attribute_name}']
        except KeyError:
            pass
        try: 
            limits[1] = self.params[f'max_{attribute_name}']
        except KeyError:
            pass
        if distribution == "normal":
            mean = self.params[f'mean_{attribute_name}']
            sd = self.params[f'sd_{attribute_name}']
            #Search if there are limits in the parameters of the Mover
            next_day = 0
            attribute_value = np.random.normal(mean, sd)
            if ((attribute_name == "resting_end_time") 
                or ((attribute_type in ["datetime_fixed", "datetime_variable"]) and (attribute_value >= 24))):
                next_day = 1
                while attribute_value >= 24:
                    attribute_value = attribute_value - 24
            i = 0
            while attribute_value < limits[0] or attribute_value > limits[1]:
                attribute_value = np.random.normal(mean, sd)
                i += 1
                if i > 100:
                    stuck_limit = limits[0] if attribute_value < limits[0] else limits[1]
                    attribute_value = stuck_limit
                    warnings.warn(f"Attribute {attribute_name} is stuck at {stuck_limit}, assigning it to {stuck_limit}.", RuntimeWarning)
            if ((attribute_name == "resting_end_time") 
                or ((attribute_type in ["datetime_fixed", "datetime_variable"]) and (attribute_value >= 24))):
                    next_day = 1
                    while attribute_value >= 24:
                        attribute_value = attribute_value - 24
            #Return based on the time
            if attribute_type in ["datetime_variable", "datetime_fixed", "timedelta"]:
                attribute_value = self.__proper_time_type(attribute_value, attribute_type, next_day)
        elif distribution == "uniform":
            attribute_value = np.random.uniform(limits[0], limits[1])
        return attribute_value
    
    def _check_activity(self) -> None:
        """Check if the agent has finished his activity"""
        if self.model.data['datetime'] >= self.data['activity_end_time']:
            self.data['status'] = "free"
            self.data['activity_end_time'] = None
            
    def _prepare_to_move(self) -> None:
        """It computes a path and changes his status to transport if the mover has a destination he is not already on the way."""
        if (self.data['step_in_path'] is None 
            and self.data['destination']['id'] is not None):
            if self.data['car'] & (distance(self.pos, self.data['destination']['position']) >= self.params['car_use_threshold']):
                network = self.model.space.roads
                self.data['network'] = "roads"
            else:
                network = self.model.space.public_transport
                self.data['network'] = "public_transport"
            self.data['path'] = self.__path_select(network)
            if not self.data['path'] is None:
                self.data['status'] = "transport"
            
    def _move(self) -> None:
        '''The method moves the mover across the path or, if he has arrived at the destination, it changes his status depending on the destination. 
        For activities, a random activity countdown is initiated.
        '''
        if self.data['status'] == "transport":
            if self.data['network'] == "roads":
                network_nodes = self.model.space.roads_nodes
            elif self.data['network'] == "public_transport":
                network_nodes = self.model.space.public_transport_nodes
            #Adds visits of Workers or PoliceAgents to the neighborhood
            if 'work_id' in self.data or 'policeman' in self.data: #Avoided isinstance for module circularity
                neighborhood_id = self.model.space.find_neighborhood_by_pos(self.geometry)
                if (self.data['last_neighborhood'] != neighborhood_id) & (neighborhood_id is not None):
                    today = str(self.model.data['datetime'].date())
                    if 'work_id' in self.data:
                        column = today + '_visits'
                    elif 'policeman' in self.data:
                        column = today + '_police'
                    self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'][column].astype(float)
                    self.model.data['info_neighborhoods'].loc[(0, neighborhood_id), column] += 1
                    self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'][column].astype(pd.SparseDtype(float, np.nan))
                    self.data['last_neighborhood'] = neighborhood_id
            #If not arrived at destination
            if self.data['step_in_path'] < len(self.data['path']):
                next_node = self.data['path'][self.data['step_in_path']]
                self.geometry = network_nodes.loc[next_node, 'geometry']
                self.data['step_in_path'] += 1
            #If arrived at destination
            elif self.data['step_in_path'] == len(self.data['path']):
                self.geometry = self.model.space.buildings.at[self.data['destination']['id'], 'geometry'].centroid
                self.data['step_in_path'], self.data['path'] = (None, None)
                if self.data['destination']['name'] in ['day_act', 'night_act']:
                    self.data['activity_end_time'] = self._gen_attribute('activity_end', attribute_type="timedelta")
                    self.data['status'] = "busy"
                else:
                    self.data['status'] = self.data['destination']['name']
                self.data['destination'] = {'id' : None, 'name' : None, 'node' : None}
                self.data['network'] = None

    def __path_select(self, network) -> list[mesa.space.FloatCoordinate]:
        """It tries to find the cached path in "outputs\_path_cache_result.pkl", otherwise computes it. 
        Finally it finds the nodes for each step according to the mover_speed specified in params 

        Returns
        ----------
            path (list[mesa.space.FloatCoordinate]) -- list of all nodes the mover will have to go through at each step
        """
        def __find_path(self) -> None:
            nearest_node = ox.distance.nearest_nodes(network, self.geometry.x, self.geometry.y)
            destination_node = ox.distance.nearest_nodes(network, self.data['destination']['position'].x, self.data['destination']['position'].y)
            self.data['step_in_path'] = 0
            if (
                cached_path := self.model.space.get_cached_path(
                    network_name = self.data['network'],
                    source_node = nearest_node,
                    target_node = destination_node
                )
            ) is not None:
                path = cached_path
            else:
                path = ox.distance.shortest_path(G = network, 
                                                orig = nearest_node,
                                                dest = destination_node,
                                                weight = 'travel_time',
                                                cpus=None)
                if path is None:                
                    path = ox.distance.shortest_path(G = network, 
                                                orig = destination_node,
                                                dest = nearest_node,
                                                weight = 'travel_time',
                                                cpus=None)
                if path is not None:
                    path.reverse()
                    self.model.space.cache_path(
                        network_name = self.data['network'],
                        source_node = nearest_node,
                        target_node = destination_node,
                        path=path,
                    )   
            return path
        path = __find_path(self)
        i = 0
        while path is None:
            self.go_to(self.data['destination']['name'])
            path = __find_path(self)
            i += 1
            if i > 10:
                warnings.warn(f"Path cannot be found between agent current position and {self.data['destination']['name']}. Assigning position of agent to destination.", RuntimeWarning)
                self.geometry = self.model.space.buildings.at[self.data['destination']['id'], 'geometry'].centroid
                self.data['destination'] = {'id' : None, 'name' : None, 'node' : None}
                return 
        #Converting path nodes to path points
        path = self.__redestribute_path_vertices(path)
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
      
        if self.data['network'] == "roads":
            network_nodes = self.model.space.roads_nodes
            network_edges = self.model.space.roads_edges
        elif self.data['network'] == "public_transport":
            network_nodes = self.model.space.public_transport_nodes
            network_edges = self.model.space.public_transport_edges
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
                        warnings.warn(f"Path cannot be found between agent current position and {self.data['destination']['name']}. Assigning position of agent to destination.", RuntimeWarning)
                        self.geometry = self.model.space.buildings.at[self.data['destination']['id'], 'geometry'].centroid
                        self.data['destination'] = {'id' : None, 'name' : None, 'node' : None}
                        return None
            edges_in_path = [get_edges_rows(edge) for edge in edges_in_path]
            if any(isinstance(edge, type(None)) for edge in edges_in_path):
                return
            edges_in_path = gpd.GeoDataFrame(edges_in_path)
            edges_in_path.loc[:, 'cumulative_travel_time'] = edges_in_path['travel_time'].cumsum()
            edges_in_path.loc[:, 'cumulative_travel_time'] = edges_in_path['cumulative_travel_time']/(self.model.model_params['len_step']*60)
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
   
    def __proper_time_type(self, adj_time : list, attribute_type : str, next_day : int) -> datetime:
        rounded_minutes = self.model.model_params['len_step'] * round(adj_time * 60 / self.model.model_params['len_step'])
        if attribute_type == "datetime_fixed":
            proper_time = [math.floor(rounded_minutes/60), rounded_minutes%60]
            #TODO: keep only this one
            if proper_time[0] == 24:
                proper_time[0] = 0
            return proper_time
        elif attribute_type == "datetime_variable":
            _minutes = timedelta(minutes = rounded_minutes)
            return self.model.data['datetime'].replace(day = self.model.data['datetime'].day+next_day,
                                                       hour = _minutes.seconds//3600, 
                                                       minute = (_minutes.seconds//60)%60)
        elif attribute_type == "timedelta":
            return self.model.data['datetime'] + timedelta(minutes = rounded_minutes)
                       
 