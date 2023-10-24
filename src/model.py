import os
import time # To measure the time of the loading of the files
import sys # To get a class from a string
import networkx as nx
import uuid
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import mesa
from shapely.geometry import Point
import osmnx as ox
from scipy.stats import skewnorm

import datetime as dt  # To keep track of the time of the simulation

from src.agents.worker import Worker
from src.agents.mover import Mover
from src.agents.criminal import Pickpocket, Robber
from src.agents.police_agent import PoliceAgent
from src.space.city import City
from typing import Any
from random import getrandbits


class StreetCrime(mesa.Model):
    
    Mover.walking_speed = 1.6 #m/s
    resident_mean_resting_start_time: int = 21
    resident_sd_resting_start_time: int = 2
    resident_mean_resting_end_time: float = 7.5
    resident_sd_resting_end_time: float = 0.83
    worker_mean_work_start_time: int = 8
    worker_sd_work_start_time: int = 2
    worker_min_work_start_time: int = 5
    worker_mean_work_end_time: int = 18
    worker_sd_work_end_time: int = 2
    worker_max_work_end_time: int = 21
    worker_mean_self_defence: float = 0.5
    worker_sd_self_defence: float = 0.17
    worker_p_information: float = 0.5
    criminal_opportunity_awareness: int = 150
    criminal_crowd_effect: float = 0.01
    criminal_p_information: float = 1
    policeagent_p_information: float = 1
    
    def __init__(self, 
                 space : City,
                 p_agents = {
                    Pickpocket: 0.07,
                    PoliceAgent: 0.05,
                    Robber: 0.03,
                    Worker: 0.85},
                 days: int = 7,
                 len_step: int = 10,
                 start_datetime: dt.datetime = dt.datetime(2020, 1, 1, 5, 30),
                 num_movers: int = 100,
                 day_act_start: int = 8,
                 day_act_end: int = 19,
                 **kwargs):
        
        self._check_parameters_validity(p_agents)
        
        self.space = space
        self.space.buildings['yesterday_visits'] = 1
        self.space.buildings['yesterday_crimes'] = 1
        self.space.buildings['yesterday_police'] = 1
        #TODO: Add mean_income to neighborhoods processing
        self.space.buildings['mean_income'] = 1
        self.space.buildings['run_visits'] = 1
        self.space.buildings['run_police'] = 1
        self.space.buildings['run_crimes'] = 1
        self.days = days
        self.len_step = len_step
        self.datetime = start_datetime
        self.num_movers = num_movers
        self.day_act_start = day_act_start
        self.day_act_end = day_act_end
        self.seed = getrandbits(128)
        self.run_id = uuid.UUID(int=self.seed, version=4)
        self.n_steps = int(self.days * 24 * 60 / self.len_step)
        self.schedule = mesa.time.RandomActivation(self)
        if p_agents is not None:
            self.p_agents = p_agents
            self.p_agents['Worker'] = 1 - sum(p_agents.values())
            self.movers = self._create_movers()
            self.agents_info = self._create_info_neighborhoods()
        #TODO: implement model_info
        self.model_info = pd.DataFrame(
            0,
            index = self.space.neighborhoods.index,
            columns = ['yesterday_visits', 'today_visits', 'run_visits',
                      'yesterday_crimes', 'today_crimes', 'run_crimes',
                      'yesterday_police', 'today_police', 'run_police'],
        )
        #TODO: implement agents_info
        self.crimes = gpd.GeoDataFrame(columns=['step', 'datetime', 'geometry', 'neighborhood',
                                        'victim', 'criminal', 'witnesses', 'successful'])
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        

    def step(self) -> None:
        """Advance the model by one step, updating agents_info at midnight each day and executing the step method of each agent.
        """
        # Advance time
        self.datetime = self.datetime + \
            timedelta(minutes=self.len_step)
                    
        # Only at midnight of each day after the first one, update information on crimes and visits of the previous day
        if (self.datetime.day > 1 and self.datetime.hour == 0 and self.datetime.minute == 0):
            self._update_information()
            self.space.save_cache_files(['roads', 'public_transport'])
        
        #Execute the step of the agents   
        self.movers.apply(lambda x: x.agent.step(), axis=1)
        
        # Randomly compute step for each agent
        #self.schedule.step()

    def get_data(self) -> dict[str, pd.DataFrame]:
        """
        Returns a dictionary with dataframes of the data of the model, the parameters  and the agents.

        
        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary with dataframes of the data of the model, the parameters  and the agents.
        """
        data = pd.DataFrame([{
            'seed' : self.seed,
            'run_id': self.run_id,
            'model_params' : self.model_params,
            'agents_params' : self.agents_params,
            'movers': self.movers,
            'model_params': None,
            'agents_params': None
        }]).set_index('run_id')
        if isinstance(self.model_params, dict):
            data.at[self.run_id, 'model_params'] = pd.DataFrame([self.model_params])
        else:
            data.at[self.run_id, 'model_params'] = pd.DataFrame([self.model_params.value])
        if isinstance(self.agents_params, dict):
            data.at[self.run_id, 'agents_params'] = pd.DataFrame([self.agents_params])
        else:
            data.at[self.run_id, 'agents_params'] = pd.DataFrame([self.agents_params.value])
        return data

    #TODO: implement agents_params, model_params checks
    def _check_parameters_validity(self, p_agents) -> None:
        p_tot = 0
        if p_agents:
            for agent, p in p_agents.items():
                if p < 0 or p > 1:
                    raise ValueError(f"Proportion of {agent} should be between 0 and 1")
                else:
                    p_tot += p
                if p_tot > 1:
                    raise ValueError(f"Sum of proportions of agents should be less than 1")                    
    
    def _update_information(self) -> None:
        """Updates 'yesterday_crimes' and 'yesterday_visits' columns of the self.buildings with data
        collected in today + '_crimes' and today + '_visits' columns of the self.neighborhoods dataframe.
        """
        self.movers_info.loc[:, 'run_visits'] += self.movers_info.loc[:, 'today_visits'] - 1
        self.movers_info.loc[:, 'run_crimes'] += self.movers_info.loc[:, 'today_crimes'] - 1
        self.movers_info.loc[:, 'run_police'] += self.movers_info.loc[:, 'today_police'] - 1
        self.movers_info.loc[:, 'yesterday_visits'] = self.movers_info.loc[:, 'today_visits'] 
        self.movers_info.loc[:, 'yesterday_crimes'] = self.movers_info.loc[:, 'today_crimes'] 
        self.movers_info.loc[:, 'yesterday_police'] = self.movers_info.loc[:, 'today_police'] 
        self.movers_info.loc[:, 'yesterday_visits'] -= 1
        self.movers_info.loc[:, 'yesterday_crimes'] -= 1
        self.movers_info.loc[:, 'yesterday_police'] -= 1
        self.movers_info.update(self.info_neighborhoos)
        self.space.buildings.drop(['yesterday_visits', 
                             'yesterday_crimes',
                             'yesterday_police',
                             'run_visits', 
                             'run_crimes',
                             'run_police'], inplace=True, axis='columns')
        copy =self.movers_info.copy()
        copy = copy[['yesterday_visits', 
                    'yesterday_crimes',
                    'yesterday_police',
                    'run_visits', 
                    'run_crimes',
                    'run_police']]
        self.space.buildings = self.space.buildings.merge(
            copy, left_on='neighborhood', right_index = True)
        
        #Add today columns
        self.movers_info = self.__set_date_columns(self.datetime.date(), self.movers_info)
        
        self.movers_info = self.movers_info.astype(pd.SparseDtype(float, np.nan))

    def _create_movers(self) -> pd.DataFrame():
        """Creates the movers of the model and adds them to the schedule and the space
        
        Parameters
        ----------
        num_movers : int
            The number of movers to create
        movers : dict[type[Mover], float]
            The dictionary of movers to create. The keys are the types of movers and the values are the percentages of each mover type to create.
        """
        start_time = time.time()
        print("Creating movers: ...")
        
        movers = pd.DataFrame(columns = ['id', 'type', 'agent']) #TODO: implement all columns

        movers['id'] = [uuid.uuid4().int for _ in range(self.num_movers)]
        movers_type = np.array([])
        for mover, p in self.p_agents.items():
            movers_type = np.concatenate((movers_type, np.full(int(self.num_movers*p), mover)))
        movers['type'] = movers_type
        movers['agent'] = movers.apply(lambda x: x.type(unique_id = x.id, 
                                                        model = self, 
                                                        geometry=None, 
                                                        crs=self.space.crs), axis=1)
        movers['agent'].apply(self.schedule.add)
        
        self.space.add_agents(movers['agent'].to_list())
        
        print("Created movers: " + "--- %s seconds ---" % (time.time() - start_time))
        return movers
    
    def _create_info_neighborhoods(self) -> gpd.GeoDataFrame:
        idx = pd.MultiIndex.from_product([self.movers.id, self.space.neighborhoods.index], names=['mover_id', 'neighborhood_id'])
        columns = ['yesterday_visits', 'today_visits', 'run_visits', 
                   'yesterday_crimes', 'today_crimes', 'run_crimes',
                   'yesterday_police', 'today_police', 'run_police']
        agents_info = pd.DataFrame(index = idx, columns = columns)
        
        #TO KEEP EVERY DAY (Adding today columns)
        #agents_info = self.__set_date_columns(self.datetime.date(), agents_info)
        
        return agents_info

    def __set_date_columns(self, date : dt.date, df : pd.DataFrame) -> pd.DataFrame:
        date = str(date)
        df.loc[0, date + '_visits'] = 1
        df.loc[0, date + '_crimes'] = 1
        df.loc[0, date + '_police'] = 1
        df[date + '_visits'] = df[date + '_visits'].astype(pd.SparseDtype(float, np.nan))
        df[date + '_crimes'] = df[date + '_crimes'].astype(pd.SparseDtype(float, np.nan))
        df[date + '_police'] = df[date + '_police'].astype(pd.SparseDtype(float, np.nan))
        return df
        
            
        