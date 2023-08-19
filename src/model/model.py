import uuid
from functools import partial
from fiona.errors import DriverError

from typing import Any
from datetime import timedelta, datetime

import pandas as pd
import geopandas as gpd
import mesa
import mesa_geo as mg
from shapely.geometry import Point

import datetime as dt  # To keep track of the time of the simulation

from src.agent.mover import Mover
from src.agent.resident import Resident
from src.agent.worker import Worker
from src.agent.criminal import Criminal
from src.agent.police_agent import PoliceAgent
from src.space.city import City
from src.space.road_network import RoadNetwork

import os.path
import time

import ast

import sys

current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]
    current_directory = parent_directory


# TODO: Use multithreading/multiprocessing for agent to find the shortest path in parallel
# TODO: Round times to the nearest timestep
# TODO: fix the generation of the resting time for the workers on the first day

class StreetCrime(mesa.Model):

    def __init__(
        self,
        files: dict[str, str] = {
            "roads_file": r"data\processed\roads.shp",
            "neighborhoods_file": r"data\processed\neighborhoods.shp",
            "buildings_file": r"data\processed\buildings.shp",
        },
        params: dict[str, Any] = {
            'crs': "epsg:7791",
            'n_steps': 50,
            'len_step': 30,  # minutes
            'p_police': 0,
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            "num_movers": 50,
            'movers': {'Worker': 0.8,
                       'Criminal': 0.1,
                       'PoliceAgent': 0.1}
        }
    ) -> None:
        super().__init__()
        city_dfs = self._load_files(files)
        self.space = City(crs=params['crs'],
                          roads_df=city_dfs['roads_df'], neighborhoods_df=city_dfs['neighborhoods_df'], buildings_df=city_dfs['buildings_df'])
        self.params = params
        self.data = {
            'step_counter': 0,
            'crimes': gpd.GeoDataFrame(columns=['step', 'datetime', 'geometry', 'neighborhood'
                                                'victim', 'criminal', 'witnesses', 'successful']),
            'datetime': params['start_datetime']
        }
        self.schedule = mesa.time.RandomActivation(self)
        self.create_movers()
        # self.datacollector = mesa.DataCollector(
        #    model_reporters=dict(zip(['Step Counter', 'Crimes', 'datetime'], self.data.values()))
        #    agent_reporters={
        #        "status": "status",
        #        "position": "geometry",
        #        "path": "path",
        #        "work_start_time": "work_start_time",
        #        "work_end_time": "work_end_time",
        #        "resting_start_time": "resting_start_time",
        #        "resting_end_time": "resting_end_time",
        #        "activity_end_time": "activity_end_time",
        #    })
        # self.datacollector.collect(self)

    # Loading files
    def _load_files(self, files: dict[str, str]) -> dict[str, gpd.GeoDataFrame]:
        city_dfs = {}

        # Roads
        start_time = time.time()
        roads_df = gpd.read_file(os.path.join(
            parent_directory, files["roads_file"]))
        city_dfs['roads_df'] = roads_df
        # Loading roads: --- 49.20670223236084 seconds ---
        print("Loading roads: " + "--- %s seconds ---" %
              (time.time() - start_time))

        # Neighborhoods
        neighborhoods_df = gpd.read_file(os.path.join(
            parent_directory, files["neighborhoods_file"]))
        neighborhoods_df.set_index('id', inplace=True)
        # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
        neighborhoods_df = neighborhoods_df.assign(
            today_visits=1, today_crimes=1)
        city_dfs['neighborhoods_df'] = neighborhoods_df

        # Buildings
        # TODO: If later we do the get nearest node when calculating the path, we don't need to have the entrance already
        buildings_df = gpd.read_file(os.path.join(
            parent_directory, files["buildings_file"]))
        buildings_df.set_index('id', inplace=True)
        buildings_df[['home', 'day_act', 'night_act']] = buildings_df[[
            'home', 'day_act', 'night_act']].astype(bool)
        buildings_df['neighborho'] = buildings_df['neighborho'].astype(int)
        buildings_df['position'] = buildings_df['position'].apply(
            ast.literal_eval)
        buildings_df['entrance'] = buildings_df['entrance'].apply(
            ast.literal_eval)
        # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
        buildings_df = buildings_df.assign(
            yesterday_visits=1, yesterday_crimes=1)
        city_dfs['buildings_df'] = buildings_df
        return city_dfs

    def create_movers(self) -> None:
        movers = []
        for mover_type in self.params['movers'].keys():
            class_type = getattr(sys.modules[__name__], mover_type)
            for _ in range(int(self.params['movers'][mover_type]*self.params['num_movers'])):
                mover = class_type(
                    unique_id=uuid.uuid4().int,
                    model=self,
                    geometry=None,
                    crs=self.space.crs)
                self.schedule.add(mover)
                movers.append(mover)
        if len(movers) > 0:
            self.space.add_agents(movers)

    def step(self) -> None:
        # Only at 2AM of each day, update information on crimes and visits of the previous day
        if (self.data['datetime'].day > 1 and self.data['datetime'].hour == 2 and self.data['datetime'].minute == 0):
            self.space.update_information()
        # Advance time
        self.data['datetime'] = self.data['datetime'] + \
            timedelta(minutes=self.params['len_step'])
        # Randomly compute step for each agent
        self.schedule.step()
        # Collect data
        # self.datacollector.collect(self)


# Running the model
Milan = StreetCrime()
for _ in range(Milan.params['n_steps']):
    Milan.data['step_counter'] += 1
    Milan.step()

Milan_df_agents = Milan.datacollector.get_agent_vars_dataframe()
Milan_df_model = Milan.datacollector.get_model_vars_dataframe()

Milan_df_agents.to_csv(os.path.join(
    parent_directory, r"outputs\run_agents.csv"))
