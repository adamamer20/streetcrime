from src.agents.resident import Resident
from mesa import Model
from shapely.geometry import Point
from pyproj import CRS
import numpy as np

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
    act_decision_rule : str = "buildings, weights = 1/df.geometry.distance(agent.geometry)*1/(df.yesterday_crimes+1) * 1/(df.run_crimes+1)"
    p_information : float = 0.5
    mean_work_start_time : float = 8
    sd_work_start_time : float = 2
    limits_work_start_time : list[float] = [5, 24]
    mean_work_end_time : float = 18
    sd_work_end_time : float = 2
    limits_work_end_time : list[float] = [0, 24]
    mean_self_defence : float = 0.5
    sd_self_defence : float = 0.3
            
    def __init__(self, 
                 unique_id: int, 
                 model: Model, 
                 geometry: Point, 
                 crs: CRS,
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
                 limits_resting_end_time : list[float] = None,
                 mean_work_start_time : float = None,
                 sd_work_start_time : float = None,
                 limits_work_start_time : list[float] = None,          
                 mean_work_end_time : float = None,
                 sd_work_end_time : float = None,
                 limits_work_end_time : list[float] = None,
                 mean_self_defence : list[float] = None,
                 sd_self_defence : list[float] = None,
                 ) -> None:

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
                    p_information,
                    mean_resting_start_time,
                    sd_resting_start_time,
                    mean_resting_end_time,
                    sd_resting_end_time,
                    car_income_threshold,
                    home_decision_rule,
                    limits_resting_start_time,
                    limits_resting_end_time)

        if mean_work_start_time is not None:
            self.mean_work_start_time = mean_work_start_time
        if sd_work_start_time is not None:
            self.sd_work_start_time = sd_work_start_time
        if limits_work_start_time is not None:
            self.limits_work_start_time = limits_work_start_time
        if mean_work_end_time is not None:
            self.mean_work_end_time = mean_work_end_time
        if sd_work_end_time is not None:
            self.sd_work_end_time = sd_work_end_time
        if limits_work_end_time is not None:
            self.limits_work_end_time = limits_work_end_time
        if mean_self_defence is not None:
            self.mean_self_defence = mean_self_defence
        if sd_self_defence is not None:
            self.sd_self_defence = sd_self_defence
            
        self.work_time = [
            self._gen_attribute(limits = self.limits_work_start_time,
                                attribute_type = 'datetime_fixed',
                                mean = self.mean_work_start_time,
                                sd = self.sd_work_start_time),
            self._gen_attribute(limits = self.limits_work_end_time,
                                attribute_type = 'datetime_fixed',
                                mean = self.mean_work_end_time,
                                sd = self.sd_work_end_time)
            ]
        
        self.limits_resting_start_time = [
            max(self.limits_resting_start_time[0], self.work_time[1][0]+self.work_time[1][1]/60),
            max(self.limits_resting_start_time[1], self.work_time[0][0]+self.work_time[0][1]/60)
        ]
        
        self.limits_resting_end_time = [
            min(self.limits_resting_end_time[0], self.work_time[0][0]+self.work_time[0][1]/60),
            min(self.limits_resting_end_time[1], self.work_time[1][0]+self.work_time[1][1]/60)
        ]
        
        self.work = self.model.space.get_random_building()
        self.defence = self._gen_attribute(limits = [0, 1],
                                           attribute_type='float',
                                           distribution = 'normal',
                                           mean = self.mean_self_defence,
                                           sd = self.sd_self_defence)
                        
        self.crime_attractiveness = self._gen_crime_attractiveness()
                
    def step(self) ->  None:
        """The step method of the Worker. Gets daily information about crimes in the city and goes to work if it's working time."""
        #Only afther the first day, at midnight, update information
        if (self.model.datetime.day > 1 
            and self.model.datetime.hour == 0 
            and self.model.datetime.minute == 0):
            self.update_info('crimes')
            
        #If it's time to go to work, go to work
        if (self.model.datetime.replace(hour = self.work_time[1][0], 
                                        minute = self.work_time[1][1]) >= 
            self.model.datetime >= 
            self.model.datetime.replace(hour = self.work_time[0][0], 
                                        minute = self.work_time[0][1])):
            if self.status not in ["work", "transport"]:
                self.go_to('work')
        else:
            if self.status == "work":
                self.status = "free"
        super().step()
    
    def _gen_crime_attractiveness(self):
        crime_attractiveness = self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
        while crime_attractiveness < 0 or crime_attractiveness > 1:
            crime_attractiveness =  self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
        return crime_attractiveness