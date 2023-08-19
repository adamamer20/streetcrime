from typing import List #Type hinting
import numpy as np 
import math 
import datetime as dt
from src.agent.resident import Resident
import pandas as pd

class Worker(Resident):
    """The Worker class is a subclass of Resident. It is used to represent the workers in the simulation.
    Workers are assigned a home based on the proportion of population in each neighborhood and have a fixed work place and working hours.
    Everyday, they go to work and do activities if they don't have to work or rest They choose activities based on recent crimes in the neighborhood.
    
    Arguments:
        unique_id (int): The unique id of the Worker.
        model (mesa.Model): The model of the simulation where the Worker is used\model\model.py
        geometry (shapely.geometry): The geometry of the Worker. In this case, Point. Home when first generated.
        crs (str): The crs of the Worker. Same as the model.

    Attributes:
        model (mesa.Model): The model of the simulation where the Worker is used\model\model.py
        geometry (shapely.geometry.Point): The geometry of the Worker. Home when first generated.
        crs (str): The crs of the Worker. Same as the model.
        params (dict[str, float]): The parameters of the Worker chosen when creating the model. See src\model\model.py
        _data (dict[str, int or List(int) or float])
            -- The data of the Resident. It contains:
                work_id (int): The id of the building where the Worker works. Assigned randomly when the Worker is created.
                work_start_time (List[int]): The time when the Worker starts working. [h, m]
                work_end_time (List[int]): The time when the Worker ends working. [h, m]
                self_defence (float): The probability of the Worker to defend himself when attacked. Generated when the Worker is created. 
                'crimes_neighborhood'(pd.DataFrame): The dataframe containing the number of crimes known to the resident in each neighborhood in the previous day.      
    """
    attributes : dict[str or int or List(int) or float] = {
        'work_id' : "gen_attribute('self_defence')",
        'work_start_time' : "gen_attribute('work_start_time', attribute_type = 'datetime_fixed')",
        'work_end_time' : "gen_time(time_type='work_end', attribute_type = 'datetime_fixed')",
        'self_defence' : "gen_attribute('self_defence')",
        'crimes_neighborhood' = pd.DataFrame()
        }
    
    params: dict[str, float] = {
        "mean_work_start_time": 8,
        "sd_work_start_time": 2,
        "mean_work_end_time": 17,
        "sd_work_end_time": 2,
        "mean_self_defence": 0.5,
        "sd_self_defence": 0.17,
        "p_information": 0.5
        },

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)
            
    def step(self) ->  None:
        #Get information about yesterday at 3AM in the morning
        if (self.model.data['datetime'].day > 0 and self.model.data['datetime'].hour == 3 and self.model.data['datetime'].minute == 0):
            self._get_yesterday_info(info_type='crimes')
        #If it's time to go to work, go to work
        if (self.model.data['datetime'].replace(hour = self.data['work_end_time'][0], minute = self.data['work_end_time'][1]) >= 
            self.model.data['datetime'] >= 
            self.model.data['datetime'].replace(hour = self.data['work_start_time'][0], minute = self.data['work_start_time'][1])) and (self.status != "work"):
            self.go_to('work')
        super(Resident, self).step()
