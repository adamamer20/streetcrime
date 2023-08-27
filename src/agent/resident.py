#Type hinting
import mesa
import pyproj
from shapely.geometry import Point
from datetime import time
from src.agent.informed_mover import InformedMover

class Resident(InformedMover):
    """The Resident Class is a subclass of InformedMover. With respect to InformedMover, it has a home generates a resting timeframe every dat.
        
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
        - home_id : int : self.model.space.get_random_building(self, 'home')
            -- The id of the building where the Resident lives. Assigned randomly when the Resident is created.
        - resting_start_time : datetime.datetime : self.gen_attribute('resting_start_time', attribute_type = 'datetime_variable')
            -- The time when the Resident starts resting.
        - resting_end_time : datetime.datetime : self.gen_attribute('resting_end_time', attribute_type = 'datetime_variable')
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
    attributes: dict[str, int or list(dt.datetime)] = {
        'home_id' : "model.space.get_random_building(function = 'home', resident = self)",
        'resting_start_time' : "gen_attribute('resting_start_time', attribute_type = 'datetime_variable')",
        'resting_end_time' : "gen_attribute('resting_end_time', attribute_type = 'datetime_variable')",
        'income' : "gen_attribute('income')"
        }
    
    params: dict[str, float] = {
        "mean_resting_start_time" : 21,
        "sd_resting_start_time" : 2,
        "mean_resting_end_time" : 7.5,
        "sd_resting_end_time" : 0.83,
        }
    
    def __init__(self, unique_id: int, model: mesa.Model, geometry: Point, crs: pyproj.CRS) -> None:
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
        super().__init__(unique_id, model, geometry, crs)
        self.data['status'] = "home"
        self.geometry = Point(self.model.space.buildings.at[self.data['home_id'], 'geometry'].centroid.coords[0])
    
    def step(self) -> None:
        """Generate resting time and proceed with Mover.step()"""
        #Generate resting time at 2pm 
        if self.model.data['datetime'].time() == time(hour = 14, minute = 0):
            self.initialize_attributes(['resting_start_time', 'resting_end_time'])
        #If it is resting time, go home
        if (self.data['resting_end_time'] >= self.model.data['datetime'] >= self.data['resting_start_time']):
            if self.data['status'] not in ['home', 'transport']:
                self.go_to('home')
        else:
            if self.data['status'] == "home":
                self.data['status'] = "free"
        super(InformedMover, self).step()
