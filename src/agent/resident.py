import mesa
import datetime as dt
from typing import List
import pyproj
from shapely.geometry import Point
from src.agent.mover import Mover

class Resident(Mover):
    """The Resident Class is a subclass of Mover. With respect to Mover, it has a home and every day the resting timeframe is generated
    
    Arguments:
        unique_id (int) -- The unique id of the Resident.
        model (mesa.Model) -- The model of the simulation where the Resident is used. See src\model\model.py
        geometry (shapely.geometry.Point) -- The point geometry of the Resident in the city.
        crs (pyproj.CRS) -- The crs of the Resident (usually same as mg.GeoSpace).
    
    Attributes:
        _data (dict[str, int or List(datetime.datetime)])
            -- The data of the Resident. It contains:
                home_id (int) -- The id of the home of the resident. Generated at the start of the function
                resting_start_time (datetime.datetime) -- The start of resting time for the Resident
                resting_end_time(datetime.datetime) -- The end of resting time for the Resident
    """
    attributes: dict[str, int or List(dt.datetime)] = {
        'home_id' : "model.space.get_random_building(self, function = 'home')",
        'resting_start_time' : "gen_attribute('resting_start_time', attribute_type = 'datetime_variable')",
        'resting_end_time' : "gen_attribute('resting_end_time', attribute_type = 'datetime_variable')",
        }
    
    params: dict[str, float] = {
        "mean_resting_start_time" : 21,
        "sd_resting_start_time" : 2,
        "mean_resting_end_time" : 7.5,
        "sd_resting_end_time" : 0.83,
        }
    
    def __init__(self, unique_id: int, model: mesa.Model, geometry: Point, crs: pyproj.CRS) -> None:
        """Generate home for Resident, sets his status and position at home

    Arguments:
        unique_id (int) -- The unique id of the Resident.
        model (mesa.Model) -- The model of the simulation where the Resident is used. See src\model\model.py
        geometry (shapely.geometry.Point) -- The point geometry of the Resident in the city.
        crs (pyproj.CRS) -- The crs of the Resident (usually same as mesa_geo.GeoSpace).
        """
        super().__init__(unique_id, model, geometry, crs)
        self.data['status'] = "home"
        self.geometry = Point(self.model.space.buildings_df.at[self.data['home_id'], "entrance"])
    
    def step(self) -> None:
        """Generate resting time and proceed with Mover.step()"""
        #Generate resting time at 2pm or at first step
        if self.model.data['datetime'].time() == dt.time(hour = 14, minute = 0):
            self.initialize_attributes('resting_start_time', 'resting_end_time')
        #If it is resting time, go home
        if (self.data['resting_end_time'] >= self.model.data['datetime'] >= self.data['resting_start_time']) and self.data['status'] != "home":
            self.go_to('home')
        super(Mover, self).step
