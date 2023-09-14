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

class StreetCrime(mesa.Model):
    def __init__(
        self,
        run_id : uuid.uuid4,
        city : dict[str, gpd.GeoDataFrame | nx.MultiDiGraph] = {
            'roads': None,
            'public_transport': None,
            'neighborhoods': None,
            'buildings': None},
        model_params = {
                    'crs': "epsg:7791",
                    'days': 7,
                    'len_step': 10,  # minutes
                    'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
                    'num_movers': 1000,
                    'day_act_start': 8,
                    'day_act_end': 19,
                    'p_agents' : {'PoliceAgent' : 0.05,
                                  'Pickpocket' : 0.07,
                                  'Robber' : 0.03},
                    },
        agents_params = {
                    'Resident': {"mean_resting_start_time": 21,
                                "sd_resting_start_time": 2,
                                "mean_resting_end_time": 7.5,
                                "sd_resting_end_time": 0.83,},
                    'Worker':   {"mean_work_start_time": 8,
                                "sd_work_start_time": 2,
                                "min_work_start_time": 5,
                                "mean_work_end_time": 18,
                                "sd_work_end_time": 2,
                                "max_work_end_time": 21,
                                "mean_self_defence": 0.5,
                                "sd_self_defence": 0.17,
                                "p_information": 0.5},
                    'Criminal': {"opportunity_awareness": 150,
                                "crowd_effect": 0.01,
                                "p_information": 1},
                    'PoliceAgent': {"p_information": 1}}) -> None:
        super().__init__()
        self.run_id = run_id
        self.model_params = model_params
        self.model_params['n_steps'] = int(model_params['days'] * 24 * 60 / model_params['len_step'])
        self.model_params['p_agents']['Worker'] = 1 - sum(model_params['p_agents'].values())
        self.agents_params = agents_params
        self.data = {
            'step_counter': 0,
            'crimes': gpd.GeoDataFrame(columns=['step', 'datetime', 'geometry', 'neighborhood',
                                                'victim', 'criminal', 'witnesses', 'successful']),
            'info_neighborhoods' : None,
            'datetime': model_params['start_datetime'],
        }
        self.space = City(
            directory = city['directory'],
            crs=model_params['crs'],
            model=self,
            roads_nodes=city['roads_nodes'],
            roads_edges=city['roads_edges'],
            roads=city['roads'],
            public_transport_nodes =city['public_transport_nodes'],
            public_transport_edges = city['public_transport_edges'],
            public_transport = city['public_transport'],
            neighborhoods=city['neighborhoods'], 
            buildings=city['buildings'].copy())
        self.schedule = mesa.time.RandomActivation(self)
        movers_id = self._create_movers(self.model_params['num_movers'], self.model_params['p_agents'])
        self.data['info_neighborhoods'] = self._create_info_neighborhoods(movers_id)
        
    def step(self) -> None:
        """Advance the model by one step, updating info_neighborhoods at midnight each day and executing the step method of each agent.
        """
        # Advance time
        self.data['datetime'] = self.data['datetime'] + \
            timedelta(minutes=self.model_params['len_step'])
        # Only at midnight of each day after the first one, update information on crimes and visits of the previous day
        if (self.data['datetime'].day > 1 and self.data['datetime'].hour == 0 and self.data['datetime'].minute == 0):
            self._update_information()
            self.space.save_cache_files(['roads', 'public_transport'])
        # Randomly compute step for each agent
        self.schedule.step()

    def get_data(self) -> dict[str, pd.DataFrame]:
        """
        Returns a dictionary with dataframes of the data of the model, the parameters  and the agents.

        
        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary with dataframes of the data of the model, the parameters  and the agents.
        """
        data = pd.DataFrame([{
            'run_id': self.run_id,
            'model_data': pd.DataFrame([self.data]),
            'agents_data': pd.DataFrame([agent.data for agent in self.space.agents]),
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
           
           
    def _update_information(self) -> None:
        """Updates 'yesterday_crimes' and 'yesterday_visits' columns of the self.buildings with data
        collected in today + '_crimes' and today + '_visits' columns of the self.neighborhoods dataframe.
        """
        self.data['info_neighborhoods'] = self.data['info_neighborhoods'].astype(float)
        yesterday = str(self.data['datetime'].date() - timedelta(days=1))
        complete_info = self.data['info_neighborhoods'].xs(0)
        complete_info.loc[:, 'run_visits'] += complete_info.loc[:, yesterday + '_visits'] - 1
        complete_info.loc[:, 'run_crimes'] += complete_info.loc[:, yesterday + '_crimes'] - 1
        complete_info.loc[:, 'run_police'] += complete_info.loc[:, yesterday + '_police'] - 1
        complete_info.loc[:, 'yesterday_visits'] = complete_info.loc[:, yesterday + '_visits'] 
        complete_info.loc[:, 'yesterday_crimes'] = complete_info.loc[:, yesterday + '_crimes'] 
        complete_info.loc[:, 'yesterday_police'] = complete_info.loc[:, yesterday + '_police'] 
        complete_info.loc[:, yesterday + '_visits'] -= 1
        complete_info.loc[:, yesterday + '_crimes'] -= 1
        complete_info.loc[:, yesterday + '_police'] -= 1
        self.data['info_neighborhoods'].update(complete_info)
        self.space.buildings.drop(['yesterday_visits', 
                             'yesterday_crimes',
                             'yesterday_police',
                             'run_visits', 
                             'run_crimes',
                             'run_police'], inplace=True, axis='columns')
        copy = complete_info.copy()
        copy = copy[['yesterday_visits', 
                    'yesterday_crimes',
                    'yesterday_police',
                    'run_visits', 
                    'run_crimes',
                    'run_police']]
        self.space.buildings = self.space.buildings.merge(
            copy, left_on='neighborhood', right_index = True)
        
        #Add today columns
        self.data['info_neighborhoods'] = self.__set_date_columns(self.data['datetime'].date(), self.data['info_neighborhoods'])
        
        self.data['info_neighborhoods'] = self.data['info_neighborhoods'].astype(pd.SparseDtype(float, np.nan))

    def _create_movers(self, num_movers: int, p_movers : dict[type[Mover], float]) -> list[int]:
        """Creates the movers of the model and adds them to the schedule and the space
        
        Parameters
        ----------
        num_movers : int
            The number of movers to create
        movers : dict[type[Mover], float]
            The dictionary of movers to create. The keys are the types of movers and the values are the percentages of each mover type to create.
        """
        start_time = time.time()
        movers = []
        for mover_type in p_movers:
            class_type = getattr(sys.modules[__name__], mover_type)
            for _ in range(int(p_movers[mover_type]*num_movers)):
                mover = class_type(
                    unique_id=uuid.uuid4().int,
                    model=self,
                    geometry=None,
                    crs=self.space.crs)
                self.schedule.add(mover)
                movers.append(mover)
        if len(movers) > 0:
            movers_id = [mover.unique_id for mover in movers]
            self.space.add_agents(movers)
        
        print("Created movers: " + "--- %s seconds ---" % (time.time() - start_time))

        return movers_id
    
    def _create_info_neighborhoods(self, movers_id : list[int]) -> gpd.GeoDataFrame:
        movers_id.append(0)
        idx = pd.MultiIndex.from_product([movers_id, self.space.neighborhoods.index], names=['mover_id', 'neighborhood_id'])
        columns = ['yesterday_visits', 'run_visits', 'yesterday_crimes', 'run_crimes', 'yesterday_police', 'run_police', 'mean_income']
        info_neighborhoods = pd.DataFrame(1, index = idx, columns = columns)
        
        #Adding today columns
        info_neighborhoods = self.__set_date_columns(self.data['datetime'].date(), info_neighborhoods)
        
        #Sorting by index
        info_neighborhoods = info_neighborhoods.sort_index()
        return info_neighborhoods

    def __set_date_columns(self, date : dt.date, df : pd.DataFrame) -> pd.DataFrame:
        date = str(date)
        df.loc[0, date + '_visits'] = 1
        df.loc[0, date + '_crimes'] = 1
        df.loc[0, date + '_police'] = 1
        df[date + '_visits'] = df[date + '_visits'].astype(pd.SparseDtype(float, np.nan))
        df[date + '_crimes'] = df[date + '_crimes'].astype(pd.SparseDtype(float, np.nan))
        df[date + '_police'] = df[date + '_police'].astype(pd.SparseDtype(float, np.nan))
        return df
