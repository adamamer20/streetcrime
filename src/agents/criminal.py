import random 
import pandas as pd
import geopandas as gpd
from src.agents.resident import Resident
from src.agents.worker import Worker
from src.agents.police_agent import PoliceAgent
from datetime import timedelta
from mesa import Model
from shapely.geometry import Point
from pyproj import CRS
import numpy as np

class Criminal(Resident):
    """The Criminal class is a subclass of Resident. Criminals are assigned a home based on the proportion of population and the income (lower income = higher probability) 
    in each neighborhood and have a fixed work place and working hours. Everyday, they go to work and do activities if they don't have to work or rest. 
    They choose activities based on the most visited neighborhoods.
    
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
    attributes : dict{str, Any}
        It defines which additional attributes a Criminal class has with respect to its parent class. 
        It can be a value or a method. The method is used for the initialization of Criminal.data. 
        It contains:
        - crime_motivation : float : self._gen_attribute('crime_motivation')
            -- Affects the probability of the Criminal to be successful in his crime . 
        - info_neighborhoods : pd.DataFrame : self.update_info('visits')
            -- The dataframe containing the number of visits known to the Criminal in each neighborhood in the previous day.
    
    params : dict[str, float] 
        It contains fixed attributes or information on how the previously specified attributes are going to be generated.
        - mean_crime_motivation : float
            -- The mean probability of the Criminal to be successful in his crime. Default: 0.5
        - sd_crime_motivation : float
            -- The standard deviation of the probability of the Criminal to be successful in his crime. Default: 0.17
        - opportunity_awareness : float
            -- The distance in meters for which the Criminal can find victims to commit a crime. Default: 300
        - crowd_deterrance : float
            -- The probability of the Criminal to be successful in his crime is reduced by this factor for each additional Worker in the vicinity. Default: 0.01
        - p_information : float
        #TODO: implement
            -- The probability of the Criminal to get information on the most visited negihborhoods and on the whereabouts of the police. Default: 1
            
    data : dict[str, Any]
        The actual attributes of the Criminal instance mirrored from the attributes and params dictionaries.
    
    Methods:
    -------
    step(): The step method of the Criminal.
    commit_crime(): Checks for crime opportunities in the vicinity and commits one if conditions permit.
    
    See Also
    --------
    Resident: src/agent/resident.py
    Mover: src/agent/mover.py
    GeoAgent: mesa/geo_agent.py
    StreetCrimeModel: src/model/model.py

    """
    attributes : dict[str or int or list(int) or float] = {
        'crime_motivation' : None,
        }
    
    params: dict[str, float] = {
        "opportunity_awareness": 150,
        "crowd_effect": 0.01,
        "p_information": 1 
        } 
    
    def __init__(self, unique_id: int, model: Model, geometry: Point, crs: CRS) -> None:
        super().__init__(unique_id, model, geometry, crs)
        self.data['crime_motivation'] = self._gen_attribute('crime_motivation')
    
    def step(self) -> None:
        """The step method of the Criminal. It checks if the Criminal can commit a crime and if so, it commits one."""
        if (self.model.data['datetime'].day > 1 
            and self.model.data['datetime'].hour == 0 
            and self.model.data['datetime'].minute == 0):
            self.update_info('visits')
        if self.data['status'] == "transport":
            self.commit_crime()
        super().step()

    def commit_crime(self) -> None:
        """Checks for crime opportunities in the vicinity and commits one if conditions permit."""
        gen_close_agents = self.model.space.get_neighbors_within_distance(self, distance = self.params['opportunity_awareness']) #Meters for which the criminal can commit a crime
        close_agents = [agent for agent in gen_close_agents if isinstance(agent, Worker)] #Only workers can be victims
        possible_victims = [agent for agent in close_agents if (agent.data['status'] == "transport") 
                                    and isinstance(agent, Worker)]
        if len(possible_victims) > 0:
            crime = {
                    'step': self.model.data['step_counter'],
                    'datetime' : self.model.data['datetime'],
                    'date' : self.model.data['datetime'].date(),
                    'position' : self.geometry,
                    'neighborhood' : self.model.space.find_neighborhood_by_pos(self.geometry),
                    'victim' : None,
                    'criminal' : self.unique_id,
                    'witnesses' : len(possible_victims) - 1,
                    'victim' : None,
                    'type' : None,
                    'successful' : None,
                    'prevented' : None,
                    }
            police = [agent for agent in close_agents if isinstance(agent, PoliceAgent)]
            if len(police) > 0: #Awareness of police in the vicinity
                neighborhood_id = [self.model.space.find_neighborhood_by_pos(police[0].geometry)]
                if neighborhood_id is not None:
                    self.model.data['info_neighborhoods'].loc[:, ['yesterday_police', 'run_police']] = self.model.data['info_neighborhoods'].loc[:, ['yesterday_police', 'run_police']].astype(float)
                    self.model.data['info_neighborhoods'].at[(self.unique_id, neighborhood_id), 'yesterday_police'] += 1
                    self.model.data['info_neighborhoods'].at[(self.unique_id, neighborhood_id), 'run_police'] += 1
                    self.model.data['info_neighborhoods'].loc[:, ['yesterday_police', 'run_police']] = self.model.data['info_neighborhoods'].loc[:, ['yesterday_police', 'run_police']].astype(pd.SparseDtype(float, np.nan))
                crime['prevented'] = True
            else:
                if len(possible_victims) > 0:
                    possible_victims = sorted(possible_victims, key = lambda x: x.data['crime_attractiveness'])
                    victim = possible_victims[0]
                    crime['victim'] = victim.unique_id
                    if self.data['crime_motivation'] < victim.data['crime_attractiveness']:
                        return
                    neighborhood_id = self.model.space.find_neighborhood_by_pos(victim.geometry)
                    #If the motivation is higher than the crowd deterrance
                    if isinstance(self, Pickpocket):
                        crime['type'] = "pickpocketing"
                        if (self.data['crime_motivation'] + self.params['crowd_effect']*(len(possible_victims)-1) + random.gauss(0, 0.05)) >= np.random.uniform(0, 1):
                            crime['successful'] = True
                        else:
                            crime['successful'] = False
                    elif isinstance(self, Robber):
                        crime['type'] = "robbery"
                        if ((self.data['crime_motivation'] - self.params['crowd_effect']*(len(possible_victims)-1) + random.gauss(0, 0.05)) >= (np.random.uniform(0, 1))) and (self.data['crime_motivation'] >= victim.data['self_defence']):
                            crime['successful'] = True
                        else:
                            crime['successful'] = False
                    crime = gpd.GeoDataFrame(crime, index = [0])
                    self.model.data['crimes'] = pd.concat([self.model.data['crimes'], crime], 
                                                        ignore_index = True)
                    today = str(self.model.data['datetime'].date())
                    column = today + '_crimes'
                    #TODO: This function for both space and movers should be merged in one
                    try:
                        self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'].loc[:, column].astype(float)
                        self.model.data['info_neighborhoods'].at[(0, neighborhood_id), column] += 1
                        self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'].loc[:, column].astype(pd.SparseDtype(float, np.nan))
                    except KeyError:
                        self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'].loc[:, column].astype(float)
                        self.model.data['info_neighborhoods'][column] = 1
                        self.model.data['info_neighborhoods'].at[(0, neighborhood_id), column] += 1
                        self.model.data['info_neighborhoods'].loc[:, column] = self.model.data['info_neighborhoods'].loc[:, column].astype(pd.SparseDtype(float, np.nan))

class Pickpocket(Criminal):
    params = {
        "act_decision_rule": "1/distance * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"
    }
    
class Robber(Criminal):
    params = {
        "act_decision_rule": "1/distance * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"
    }