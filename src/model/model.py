import os.path # To get the path of the files
import time # To measure the time of the loading of the files
import sys # To get a class from a string
import ast # To convert string to list

import uuid
from typing import Any
from datetime import timedelta, datetime

import pandas as pd
import geopandas as gpd
import mesa
from shapely.geometry import Point
import osmnx as ox

import datetime as dt  # To keep track of the time of the simulation

from src.agent.worker import Worker
from src.agent.criminal import Criminal
from src.agent.police_agent import PoliceAgent
from src.space.city import City


current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]
    current_directory = parent_directory

# TODO: fix the generation of the resting time for the workers on the first day
# TODO : check that the resident implementation of resting time (initizialization) does not create problems

class StreetCrime(mesa.Model):
    def __init__(
        self,
        files: dict[str, str] = {
            "roads_file": r"data\processed\roads.graphml",
            "neighborhoods_file": r"data\processed\neighborhoods.shp",
            "buildings_file": r"data\processed\buildings.shp",
        },
        params: dict[str, Any] = {
            'crs': "epsg:7791",
            'n_steps': 300,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            "num_movers": 10,
            'movers': {'Criminal': 0.3,
                       'PoliceAgent': 0.3},
            'day_act_start': 8,
            'day_act_end': 19,}
        ) -> None:
        super().__init__()
        self.params = params
        city = self._load_files(files)
        self.space = City(crs=params['crs'],
                          model=self,
                          road_network=city['road_network'], 
                          neighborhoods=city['neighborhoods'], 
                          buildings=city['buildings'])
        params['movers']['Worker'] = 1 - sum(params['movers'].values())

        self.data = {
            'step_counter': 0,
            'crimes': gpd.GeoDataFrame(columns=['step', 'datetime', 'geometry', 'neighborhood',
                                                'victim', 'criminal', 'witnesses', 'successful']),
            'info_neighborhoods' : self.space.neighborhoods,
            'datetime': params['start_datetime'],
        }
        self.schedule = mesa.time.RandomActivation(self)
        self.create_movers()
        self.datacollector = mesa.DataCollector(
           model_reporters={
                'step_counter': "data['step_counter']",
                'datetime' : "data['datetime']"
            })
            #agent_reporters = {
            #    'destination_id' : "data['destination']['id']",
            #    'destination_node' : "data['destination']['node']",
            #    'destination_name' : "data['destination']['name']",
            #    'step_in_path' : "data['step_in_path']",
            #    'path' : "data['path']",
            #    'status' : "data['status']",
            #    'last_neighborhood' : "data['last_neighborhood']",
            #    'activity_end_time' : "data['activity_end_time']",
            #})

    def _load_files(self, files: dict[str, str]) -> dict[str, gpd.GeoDataFrame]:
        """Load the files of the city and return a dictionary of GeoDataFrames with the loaded files

        Parameters
        ----------
        files : dict[str, str]
            The files to load. The keys are the names of the files and the values are the paths to the files

        Returns
        -------
        dict[str, gpd.GeoDataFrame]
            A dictionary of GeoDataFrames with the loaded files. 
                - roads_df: The roads of the city
                - neighborhoods: The neighborhoods of the city
                - buildings: The buildings of the city
        """
        city = {}

        # Roads
        start_time = time.time()
        city['road_network'] = ox.io.load_graphml(os.path.join(parent_directory, files["roads_file"]))
        # Loading roads: --- 7.504880905151367 seconds ---
        print("Loading roads: " + "--- %s seconds ---" %
              (time.time() - start_time))

        # Neighborhoods
        neighborhoods = gpd.read_file(os.path.join(
            parent_directory, files["neighborhoods_file"]))
        neighborhoods.set_index('id', inplace=True)
        current_date = str(self.params['start_datetime'].date())
        neighborhoods = neighborhoods.assign(**{
            current_date + '_visits': 1,
            current_date + '_crimes': 1,
            current_date + '_police': 1,
            "run_visits" : 1,
            "run_crimes" : 1,
            "run_police" : 1,
        })
        neighborhoods.drop(['name', 'cap'], axis='columns', inplace=True)
        city['neighborhoods'] = neighborhoods

        # Buildings
        start_time = time.time()
        buildings = gpd.read_file(os.path.join(
            parent_directory, files["buildings_file"]))
        buildings.set_index('id', inplace=True)
        buildings[['home', 'day_act', 'night_act']] = buildings[[
            'home', 'day_act', 'night_act']].astype(bool)
        buildings.rename(columns={'neighborho': 'neighborhood', 
                                     'entrance_n' : 'entrance_node'}, inplace=True)
        buildings['neighborhood'] = buildings['neighborhood'].astype(int)
        # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
        buildings = buildings.assign(
            yesterday_visits = 1,
            run_visits = 1, 
            yesterday_crimes = 1, 
            run_crimes = 1,
            yesterday_police = 1,
            run_police = 1)
        buildings.drop('function', axis='columns', inplace=True)
        city['buildings'] = buildings
        print("Loading buildings: " + "--- %s seconds ---" %
              (time.time() - start_time))
        return city

    def create_movers(self) -> None:
        """Creates the movers of the model and adds them to the schedule and the space
        """
        movers = []
        for mover_type in self.params['movers']:
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
        """Advance the model by one step, updating info_neighborhoods at midnight each day and executing the step method of each agent.
        """
        # Advance time
        self.data['datetime'] = self.data['datetime'] + \
            timedelta(minutes=self.params['len_step'])
        # Only at midnight of each day after the first one, update information on crimes and visits of the previous day
        if (self.data['datetime'].day > 1 and self.data['datetime'].hour == 0 and self.data['datetime'].minute == 0):
            self.space.update_information()
        # Randomly compute step for each agent
        self.schedule.step()

