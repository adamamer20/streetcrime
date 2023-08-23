import math #Extract working hour
from datetime import timedelta, datetime
import copy #Necessary for immutable objects in attributes dictionary

import pyproj #Type hinting for CRS
import numpy as np #Generate working hours

import mesa
import mesa_geo as mg

import osmnx as ox

from shapely.geometry import Point, LineString

from src.space.utils import redistribute_vertices

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
    gen_attribute(attribute_name : str, attribute_type : str = "float", distribution = "normal") -> float | datetime | list[int]
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
        'last_neighborhood' : None
        }
    
    params: dict[str, float] = {
        "walking_speed": 1.9, #m/s
        "driving_speed": 7, #m/s
        "max_walking_distance" : 4000, #m
        "mean_activity_end" : 1,
        "sd_activity_end" : 0.3
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
            for params_name, params_value in class_obj.params.items():
                self.params.setdefault(params_name, params_value)
        self.initialize_attributes()
    
    @property
    def geometry(self):
        return self._geometry
    
    @geometry.setter
    def geometry(self, new_geometry):
        self._geometry = new_geometry
        self.data['geometry'] = new_geometry

    def gen_attribute(self, 
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
        if attribute_name in ["resting_start_time", "resting_end_time"] and self.model.data['datetime'] == self.model.params['start_datetime']: 
            if attribute_name == "resting_start_time":
                attribute_value = self.model.params['start_datetime']
            elif attribute_name == "resting_end_time":
                attribute_value = self.model.params['start_datetime'] + timedelta(hours = np.random.normal(1, 0.5))
        else:
            match attribute_type:
                case "float":
                    limits = [0,1]
                case "datetime_fixed" | "datetime_variable":
                    limits = [float('-inf'), float('+inf')]
                case "timedelta":
                    limits = [0, float('+inf')]
            mean = self.params[f'mean_{attribute_name}']
            sd = self.params[f'sd_{attribute_name}']
            #Search if there are limits in the parameters of the Mover
            try: 
                limits[0] = self.params[f'min_{attribute_name}']
            except KeyError:
                pass
            try: 
                limits[1] = self.params[f'max_{attribute_name}']
            except KeyError:
                pass
            next_day = 0
            attribute_value = np.random.normal(mean, sd)
            if ((attribute_name == "resting_end_time") 
                or ((attribute_type in ["datetime_fixed", "datetime_variable"]) and (attribute_value >= 24))):
                next_day = 1
                if attribute_value >= 24:
                    attribute_value = attribute_value - 24
            while attribute_value < limits[0] or attribute_value > limits[1]:
                attribute_value = np.random.normal(mean, sd)
            if ((attribute_name == "resting_end_time") 
                or ((attribute_type in ["datetime_fixed", "datetime_variable"]) and (attribute_value >= 24))):
                    next_day = 1
                    if attribute_value >= 24:
                        attribute_value = attribute_value - 24
            #Return based on the time
            if attribute_type in ["datetime_variable", "datetime_fixed", "timedelta"]:
                attribute_value = self._proper_time_type(attribute_value, attribute_type, next_day)
        return attribute_value
            
    def _proper_time_type(self, adj_time : list, attribute_type : str, next_day : int) -> datetime:
        rounded_minutes = self.model.params['len_step'] * round(adj_time * 60 / self.model.params['len_step'])
        if attribute_type == "datetime_fixed":
            return [math.floor(rounded_minutes/60), rounded_minutes%60]
        elif attribute_type == "datetime_variable":
            _minutes = timedelta(minutes = rounded_minutes)
            return self.model.data['datetime'].replace(day = self.model.data['datetime'].day+next_day,
                                                       hour = _minutes.seconds//3600, 
                                                       minute = (_minutes.seconds//60)%60)
        elif attribute_type == "timedelta":
            return self.model.data['datetime'] + timedelta(minutes = rounded_minutes)
    
    def initialize_attributes(self, attributes : list or str = None) -> None:
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

    def step(self) -> None:
        """The Mover gets a random building if he is free, computes a path with an active destination if he hasn't started travelling, moves if he is travelling.
        """        
        if self.data['status'] == "busy":
            self._check_activity()
        if self.data['destination']['id'] is None and self.data['status'] == "free":
            if (self.model.data['datetime'] >= 
                self.model.data['datetime'].replace(hour = self.model.params['day_act_start']) 
                and self.model.data['datetime'] <= 
                self.model.data['datetime'].replace(hour = self.model.params['day_act_end'])):
                self.go_to("day_act")
            else:
                self.go_to("night_act")
        self._prepare_to_move()   
        self._move()
    
    def _check_activity(self) -> None:
        """Check if the agent has finished his activity"""
        if self.model.data['datetime'] >= self.data['activity_end_time']:
            self.data['status'] = "free"
            self.data['activity_end_time'] = None
            
    def _prepare_to_move(self) -> None:
        """It computes a path and changes his status to transport if the mover has a destination he is not already on the way."""
        if (self.data['step_in_path'] is None 
            and self.data['destination']['id'] is not None):
            self.data['path'] = self.__path_select()
            self.data['status'] = "transport"
            
    def __path_select(self) -> list[mesa.space.FloatCoordinate]:
        """It tries to find the cached path in "outputs\_path_cache_result.pkl", otherwise computes it. 
        Finally it finds the nodes for each step according to the mover_speed specified in params 

        Returns
        ----------
            path (list[mesa.space.FloatCoordinate]) -- list of all nodes the mover will have to go through at each step
        """
        self.data['step_in_path'] = 0
        if (
            cached_path := self.model.space.get_cached_path(
                ox.distance.nearest_nodes(self.model.space.road_network, self.geometry.x, self.geometry.y),
                self.data['destination']['node']
            )
        ) is not None:
            path = cached_path
        else:
            path = ox.distance.shortest_path(G = self.model.space.road_network, 
                                             orig = ox.distance.nearest_nodes(self.model.space.road_network, self.geometry.centroid.x, self.geometry.centroid.y),
                                             dest = self.data['destination']['node'], 
                                             cpus=None)
            self.model.space.cache_path(
                ox.distance.nearest_nodes(self.model.space.road_network, self.geometry.centroid.x, self.geometry.centroid.y),
                self.data['destination']['node'],
                path=path,
            )
        #Converting path nodes to path points
        path = [self.model.space.roads_nodes.at[node, 'geometry'] for node in path]
        path = self.__redestribute_path_vertices(path)
        return path
    
    def __redestribute_path_vertices(self, original_path : list[mesa.space.FloatCoordinate]) -> None:
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
        # if origin and destination are the same, no need to redistribute path vertices.
        if len(original_path) > 1:
            original_path = LineString(original_path)
            if original_path.length < self.params['max_walking_distance']:
                speed = self.params['walking_speed']
            else:
                speed = self.params['driving_speed']
            redistributed_path_in_meters = redistribute_vertices(
                original_path, speed*self.model.params['len_step']*60 #From m/s to m/step
            )
            redistributed_path = list(redistributed_path_in_meters.coords)
        else:
            redistributed_path = original_path
        return redistributed_path 
                
    def _move(self) -> None:
        '''The method moves the mover across the path or, if he has arrived at the destination, it changes his status depending on the destination. 
        For activities, a random activity countdown is initiated.
        '''
        if self.data['status'] == "transport":
            #Adds visits of Workers or PoliceAgents to the neighborhood
            if 'work_id' in self.data or 'policeman' in self.data: #Avoided isinstance for module circularity
                neighborhood_id = self.model.space.find_neighborhood_by_position(self.geometry)
                if self.data['last_neighborhood'] != neighborhood_id:
                    today = str(self.model.data['datetime'].date())
                    if 'work_id' in self.data:
                        column = today + '_visits'
                    elif 'policeman' in self.data:
                        column = today + '_police'
                    try:
                        self.model.space.neighborhoods.at[neighborhood_id, column] += 1
                    except KeyError:
                        self.model.space.neighborhoods[column] = 1
                        self.model.space.neighborhoods.at[neighborhood_id, column] += 1
                    self.data['last_neighborhood'] = neighborhood_id
            #If not arrived at destination
            if self.data['step_in_path'] < len(self.data['path']):
                next_position = self.data['path'][self.data['step_in_path']]
                self.geometry = Point(next_position)
                self.data['step_in_path'] += 1
            #If arrived at destination
            elif self.data['step_in_path'] == len(self.data['path']):
                self.geometry = self.model.space.buildings.at[self.data['destination']['id'], 'geometry'].centroid
                self.data['step_in_path'], self.data['path'] = (None, None)
                if self.data['destination']['name'] in ['day_act', 'night_act']:
                    self.data['activity_end_time'] = self.gen_attribute('activity_end', attribute_type="timedelta")
                    self.data['status'] = "busy"
                else:
                    self.data['status'] = self.data['destination']['name']
                self.data['destination'] = {'id' : None, 'name' : None, 'node' : None}


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
        self.data['destination']['node'] = self.model.space.buildings.at[building_id, 'entrance_node']


                       
 