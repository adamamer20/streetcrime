from src.agents.resident import Resident
from mesa import Model
from shapely.geometry import Point
from pyproj import CRS

class Worker(Resident):
    """The Worker class is a subclass of Resident. Workers are assigned a home based on the proportion of population in each neighborhood and have a fixed work place and working hours.
    Everyday, they go to work and do activities if they don't have to work or rest. They choose activities based on recent crimes in the neighborhood.
    
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
    params : dict[str, float] 
        It contains fixed attributes or information on how attributes are going to be generated.
        - mean_work_start_time : float 
            -- The mean time when the Worker starts working. Default: 8
        - sd_work_start_time : float
            -- The standard deviation of the time when the Worker starts working. Default: 2
        - mean_work_end_time : float
            -- The mean time when the Worker ends working. Default: 18
        - sd_work_end_time : float 
            -- The standard deviation of the time when the Worker ends working. Default: 2
        - mean_self_defence : float
            -- The mean probability of the Worker to defend himself when robbed. Default: 0.5
        - sd_self_defence : float
            -- The standard deviation of the probability of the Worker to defend himself when robbed. Default: 0.17
        - p_information : float  
            -- The percentage of crimes(randomly sampled) the Worker will know about the previous day. Default: 0.5


    Attributes:
    ----------
    attributes : dict[str or int or list(int) or float]
        It defines which additional attributes a Worker class has with respect to its parent class. 
        It can be a value or a method. The method is used for the initialization of Worker.data. 
        It contains:
        - work_id : int : self.model.space.get_random_building(self, 'work)
            -- The id of the building where the Worker works. Assigned randomly when the Worker is created.
        - work_start_time : list[int] : self._gen_attribute('work_start_time', attribute_type = 'datetime_fixed')
            -- The time when the Worker starts working. [h, m]
        - work_end_time : list[int] : self._gen_attribute('work_end_time', attribute_type = 'datetime_fixed')
            -- The time when the Worker ends working. [h, m]
        - self_defence : float : self._gen_attribute('self_defence', attribute_type = 'float')
            -- Affects the probability of the Worker to defend himself when attacked. 
        - info_neighborhoods : pd.DataFrame : self.update_info(crimes)
            -- The dataframe containing the number of crimes known to the resident in each neighborhood in the previous day.      
        
    data : dict[str, Any]
        The actual attributes of the Worker instance mirrored from the attributes and params dictionaries.
    
    Methods:
    -------
    step(): The step method of the Worker.
    
    See Also
    --------
    Resident: src/agent/resident.py
    Mover: src/agent/mover.py
    GeoAgent: mesa/geo_agent.py
    StreetCrimeModel: src/model/model.py
    
    """ 
    
    attributes : dict[str or int or list(int) or float] = {
        'work_id' : "model.space.get_random_building('work', self)",
        'work_start_time' : "_gen_attribute('work_start_time', attribute_type = 'datetime_fixed')",
        'work_end_time' : "_gen_attribute('work_end_time', attribute_type = 'datetime_fixed')",
        'self_defence' : "_gen_attribute('self_defence', attribute_type = 'float', distribution = 'uniform')",
        'crime_attractiveness' : None,
        }
    
    params: dict[str, float] = {
        "mean_work_start_time": 8,
        "sd_work_start_time": 2,
        "min_work_start_time": 5,
        "mean_work_end_time": 18,
        "sd_work_end_time": 2,
        "max_work_end_time": 21,
        "p_information": 0.5,
        "car_income_threshold": 30000,
        "act_decision_rule": "1/distance * 1/(yesterday_crimes+1) * 1/(run_crimes+1)"
        }
            
    def __init__(self, unique_id: int, model: Model, geometry: Point, crs: CRS) -> None:
        super().__init__(unique_id, model, geometry, crs)
        self.data['crime_attractiveness'] = self._gen_attribute('crime_attractiveness')
        self.data['car'] = self._gen_attribute('car', attribute_type = 'bool')
        
    def step(self) ->  None:
        """The step method of the Worker. Gets daily information about crimes in the city and goes to work if it's working time."""
        #Only afther the first day, at midnight, update information
        if (self.model.data['datetime'].day > 1 
            and self.model.data['datetime'].hour == 0 
            and self.model.data['datetime'].minute == 0):
            self.update_info('crimes')
        #If it's time to go to work, go to work
        if (self.model.data['datetime'].replace(hour = self.data['work_end_time'][0], 
                                                minute = self.data['work_end_time'][1]) >= 
            self.model.data['datetime'] >= 
            self.model.data['datetime'].replace(hour = self.data['work_start_time'][0], 
                                                minute = self.data['work_start_time'][1])):
            if self.data['status'] not in ["work", "transport"]:
                self.go_to('work')
        else:
            if self.data['status'] == "work":
                self.data['status'] = "free"
        super().step()