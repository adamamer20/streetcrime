from typing import List #Type hinting
import datetime as dt #Generate working hours

import numpy as np #Generate working hours
import random 

import mesa_geo as mg

import pandas as pd

from src.agent.resident import Resident
from src.agent.worker import Worker
from src.agent.police_agent import PoliceAgent

class Criminal(Resident):
    attributes : dict[str or int or List(int) or float] = {
        'crime_motivation' : "gen_attribute('crime_motivation')",
        #'criminal_neighborhoods' : pd.DataFrame(0, columns = ['yesterday_crimes'])
        }
    params: dict[str, float] = {
        "mean_crime_motivation": 0.5,
        "sd_crime_motivation": 0.17,
        "opportunity_awareness": 300,
        "crowd_deterrance": 0.01,
        "p_information": 1 #TODO: implement
        } 

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)
    
    def step(self) -> None:
        if self.data['status'] == "transport":
            self.commit_crime()
        super().step()

    def commit_crime(self) -> None:
        gen_close_agents = self.model.space.get_neighbors_within_distance(self, distance = self.params['opportunity_awareness']) #Meters for which the criminal can commit a crime
        close_agents = [agent for agent in gen_close_agents if isinstance(agent, Worker)] #Only workers can be victims
        if any(isinstance(agent, PoliceAgent) for agent in close_agents): #Awareness of police in the vicinity
            return
        else:
            possible_victims = [agent for agent in close_agents if (agent.data['status'] == "transport") 
                                and isinstance(agent, Worker)]
            if len(possible_victims) > 0:
                #If the motivation is higher than the crowd deterrance
                if self.crime_motivation >= (self.params['crowd_deterrance'] * len(possible_victims)-1): 
                    victim = random.choice(possible_victims) #Choose a random victim
                    #Find the neighborhood where the crime is committed
                    for id, neighborhood in self.model.space.neighborhoods_df.iterrows(): #Maybe external callable function of city?
                        if neighborhood.geometry.contains(victim.geometry):
                            neighborhood_id = id
                            break
                    crime = {
                        'step': self.model.model_data['step_counter'],
                        'datetime' : self.model.data['datetime'],
                        'geometry' : victim.geometry,
                        'neighborhood' : neighborhood_id,
                        'victim' : victim.unique_id,
                        'criminal' : self.unique_id,
                        'witnesses' : len(possible_victims) - 1
                        }
                    if victim.self_defence < self.crime_motivation:
                        crime['succesful'] = True
                    else:
                        crime['succesful'] = False
                    self.model.model_data['crimes'] = self.model.model_data['crimes'].append(crime, ignore_index = True) #FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
                    self.model.space.neighborhoods_df.at[neighborhood_id, 'today_crimes'] += 1
                        