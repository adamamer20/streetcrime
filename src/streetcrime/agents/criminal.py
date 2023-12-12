import random 
import pandas as pd
import geopandas as gpd
from streetcrime.agents.resident import Resident
from streetcrime.agents.worker import Worker
from streetcrime.agents.police_agent import PoliceAgent
from datetime import timedelta
import mesa
import pyproj
from shapely.geometry import Point
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
        The model of the simulation where the Mover is used. See streetcrime/model/model.py
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
    Resident: streetcrime/agent/resident.py
    Mover: streetcrime/agent/mover.py
    GeoAgent: mesa/geo_agent.py
    StreetCrimeModel: streetcrime/model/model.py

    """
    
    '''TODO: add crime_motivation
            elif attribute_name == "crime_motivation":
                attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.income) + np.random.normal(0, 0.10)
                while attribute_value < 0 or attribute_value > 1:
                    attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.income) + np.random.normal(0, 0.10)
                return attribute_value'''
    opportunity_awareness : int = 150
    crowd_effect : float = 0.01
    p_information : float = 1
    home_decision_rule : str = "neighborhoods, weights = df.prop * (1/df.mean_income)"
    
    
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
                 limits_resting_end_time : list[float] = None,
                 opporunity_awareness = None, 
                 crowd_effect = None) -> None:
        
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

        if opporunity_awareness is not None:
            self.opportunity_awareness = opporunity_awareness
        if crowd_effect is not None:
            self.crowd_effect = crowd_effect
        
        self.crime_motivation = self._gen_crime_motivation()
    
    def step(self) -> None:
        """The step method of the Criminal. It checks if the Criminal can commit a crime and if so, it commits one."""
        #Update informations
        if (self.model.datetime.day > 1 
            and self.model.datetime.hour == 0 
            and self.model.datetime.minute == 0):
            self.update_info('visits')
        #Commit crime
        if self.status == "transport":
            self.commit_crime()
        super().step()

    def commit_crime(self) -> None:
        """Checks for crime opportunities in the vicinity and commits one if conditions permit."""
        close_agents = list(self.model.space.get_neighbors_within_distance(self, distance = self.opportunity_awareness)) #Meters for which the criminal can commit a crime
        possible_victims = [agent for agent in close_agents if (agent.status == "transport") and isinstance(agent, Worker)]
        if len(possible_victims) > 0:
            crime = {
                    'step': self.model.step,
                    'datetime' : self.model.datetime,
                    'date' : self.model.datetime.date(),
                    'position' : self.geometry,
                    'neighborhood' : self.model.space.find_neighborhood_by_pos(self.geometry),
                    'criminal' : self.unique_id,
                    'witnesses' : len(possible_victims) - 1,
                    'victim' : None,
                    'type' : None,
                    'successful' : None,
                    'prevented' : False,
                    }
            if isinstance(self, Pickpocket):
                crime['type'] = "pickpocketing"
            elif isinstance(self, Robber):
                crime['type'] = "robbery"
            police = [agent for agent in close_agents if isinstance(agent, PoliceAgent)]
            if len(police) > 0: #Awareness of police in the vicinity
                neighborhood_id = [self.model.space.find_neighborhood_by_pos(police[0].geometry)]
                if neighborhood_id is not None:
                    #TODO: has to be changed
                    #self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']] = self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']].astype(float)
                    #self.model.info_neighborhoods.at[(self.unique_id, neighborhood_id), 'yesterday_police'] += 1
                    #self.model.info_neighborhoods.at[(self.unique_id, neighborhood_id), 'run_police'] += 1
                    #self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']] = self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']].astype(pd.SparseDtype(float, np.nan))
                    pass
                crime['prevented'] = True
            else:
                if len(possible_victims) > 0:
                    possible_victims = sorted(possible_victims, key = lambda x: x.crime_attractiveness, reverse = True)
                    victim = possible_victims[0]
                    crime['victim'] = victim.unique_id
                    neighborhood_id = self.model.space.find_neighborhood_by_pos(victim.geometry)
                    #If the motivation is higher than the crowd deterrance
                    if isinstance(self, Pickpocket):
                        if (self.crime_motivation + self.crowd_effect*(len(possible_victims)-1) + random.gauss(0, 0.05)) >= np.random.uniform(0, 1):
                            crime['successful'] = True
                        else:
                            crime['successful'] = False
                    elif isinstance(self, Robber):
                        if ((self.crime_motivation - self.crowd_effect*(len(possible_victims)-1) + random.gauss(0, 0.05)) >= (np.random.uniform(0, 1))) and (self.crime_motivation >= victim.defence):
                            crime['successful'] = True
                        else:
                            crime['successful'] = False
                crime = gpd.GeoDataFrame(crime, index = [0])
                self.model.crimes = pd.concat([self.model.crimes, crime], 
                                              ignore_index = True)
                today = str(self.model.datetime.date())
                column = today + '_crimes'
                #TODO: This function for both space and movers should be merged in one
                try:
                    #self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods.loc[:, column].astype(float)
                    #self.model.info_neighborhoods.at[(0, neighborhood_id), column] += 1
                    #self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods.loc[:, column].astype(pd.SparseDtype(float, np.nan))
                    pass
                except KeyError:
                    #self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods.loc[:, column].astype(float)
                    #self.model.info_neighborhoods[column] = 1
                    #self.model.info_neighborhoods.at[(0, neighborhood_id), column] += 1
                    #self.model.info_neighborhoods.loc[:, column] = self.model.info_neighborhoods.loc[:, column].astype(pd.SparseDtype(float, np.nan))
                    pass

    def _gen_crime_motivation(self) -> float:
        crime_motivation = 1 - self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
        while crime_motivation < 0 or crime_motivation > 1:
            crime_motivation =  1 - self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
        return crime_motivation

class Pickpocket(Criminal):
    act_decision_rule : str = "buildings, weights = (1/df.geometry.distance(agent.geometry)) * df.yesterday_visits * df.run_visits * df.mean_income * (1/df.yesterday_police) * (1/df.run_police)"
    
class Robber(Criminal):
    act_decision_rule : str = "buildings, weights = (1/df.geometry.distance(agent.geometry)) * (1/df.yesterday_visits) * (1/df.run_visits) * df.mean_income * (1/df.yesterday_police) * (1/df.run_police)"