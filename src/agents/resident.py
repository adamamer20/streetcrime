#Type hinting
import mesa
import pyproj
from shapely.geometry import Point
from datetime import time
from src.agents.informed_mover import InformedMover
import datetime as dt
import numpy as np

class Resident(InformedMover):
    """The Resident Class is a subclass of InformedMover. With respect to InformedMover, it has a home generates a resting timeframe every day.
        
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
    attributes : dict[str : str]
        It defines which additional attributes a Resident class has with respect to its parent class.
        It can be a value or a method. The method is used for the initialization of the attribute in Resident.data. 
        It contains:
        - home : int : self.model.space.get_random_building(self, 'home')
            -- The id of the building where the Resident lives. Assigned randomly when the Resident is created.
        - resting_start_time : datetime.datetime : self._gen_attribute('resting_start_time', attribute_type = 'datetime_variable')
            -- The time when the Resident starts resting.
        - resting_end_time : datetime.datetime : self._gen_attribute('resting_end_time', attribute_type = 'datetime_variable')
            -- The time when the Resident ends resting. 
    
    params : dict[str, float]
        It contains fixed attributes or information on how the previously specified attributes are going to be generated.
        - mean_resting_start_time : float
            -- The mean time when the Resident starts resting. Default: 21
        - sd_resting_start_time : float
            -- The standard deviation of the time when the Resident starts resting. Default: 2
        - mean_resting_end_time : float
            -- The mean time when the Resident ends resting. Default: 7.5
        - sd_resting_end_time : float
            -- The standard deviation of the time when the Resident ends resting. Default: 0.83
         
    """
    mean_resting_start_time: float = 21
    sd_resting_start_time: float = 2
    mean_resting_end_time: float = 8
    sd_resting_end_time: float = 0.83
    car_income_threshold: float = 30000
    home_decision_rule : str = 'neighborhoods, weights = df.prop'
    limits_resting_start_time : list[float] = [0, 24]
    limits_resting_end_time : list[float] = [0, 24]

    def __init__(self, 
                 unique_id: int, 
                 model: mesa.Model, 
                 geometry: Point, 
                 crs: pyproj.CRS,
                 walking_speed : float = None,
                 driving_speed : float= None,
                 car_use_threshold : float = None,
                 sd_activity_end : float = None,
                 mean_activity_end : float = None,
                 car : bool = None,
                 act_decision_rule : str = None,
                 p_information : float = None,
                 mean_resting_start_time : float = None,
                 sd_resting_start_time : float = None,
                 mean_resting_end_time : float = None,
                 sd_resting_end_time : float = None,
                 car_income_threshold : float = None,
                 home_decision_rule : str = None,
                 limits_resting_start_time : list[float] = None,
                 limits_resting_end_time : list[float] = None) -> None:
        
        """Sets Resident status and position at home
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
        super().__init__(unique_id, 
                         model, 
                         geometry, 
                         crs, 
                         walking_speed, 
                         driving_speed, 
                         car_use_threshold, 
                         sd_activity_end, 
                         mean_activity_end,
                         car,
                         act_decision_rule,
                         p_information)

        
        if mean_resting_start_time is not None:
            self.mean_resting_start_time = mean_resting_start_time
        if sd_resting_start_time is not None:
            self.sd_resting_start_time = sd_resting_start_time
        if mean_resting_end_time is not None:
            self.mean_resting_end_time = mean_resting_end_time
        if sd_resting_end_time is not None:
            self.sd_resting_end_time = sd_resting_end_time
        if car_income_threshold is not None:
            self.car_income_threshold = car_income_threshold
        if home_decision_rule is not None:
            self.home_decision_rule = home_decision_rule
        if limits_resting_start_time is not None:
            self.limits_resting_start_time = limits_resting_start_time
        if limits_resting_end_time is not None:
            self.limits_resting_end_time = limits_resting_end_time
        
        self.resting_time = self._gen_resting_time()
        self.home = model.space.get_random_building(function = 'home',  agent = self, decision_rule = self.home_decision_rule)
        self.geometry = self.model.space.buildings.at[self.home, 'geometry'].centroid
        self.status = "home"
        self.income = self._gen_attribute(limits=None,
                                          attribute_type='float',
                                          distribution = 'skewnorm',
                                          a = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'ae'],
                                          loc = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'loce'],
                                          scale = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'scalee'])
        self.car = self.income/self.car_income_threshold+np.random.normal(0, 0.1) > 0.5
        
    def step(self) -> None:
        """Generate resting time and proceed with Mover.step()"""
        #Generate resting time at 2pm 
        if self.model.datetime.time() == time(hour = 14, minute = 0):
            self.resting_time = self._gen_resting_time()
        #If it is resting time, go home
        if (self.resting_time[1] >= self.model.datetime >= self.resting_time[0]):
            if self.status not in ['home', 'transport']:
                self.go_to('home')
        else:
            if self.status == "home":
                self.status = "free"
        super().step()
        
    def _gen_resting_time(self) -> list[dt.datetime]:
        
        resting_start_time = self._gen_attribute(limits = self.limits_resting_start_time,
                                                attribute_type = 'datetime_variable',
                                                mean = self.mean_resting_start_time,
                                                sd = self.sd_resting_start_time)
        
        resting_end_time = self._gen_attribute(limits = self.limits_resting_end_time,
                                               attribute_type = 'datetime_variable',
                                               mean = self.mean_resting_end_time,
                                               sd = self.sd_resting_end_time,
                                               next_day=True)  
        
        return [resting_start_time, resting_end_time]
    