import os.path # To get the path of the files
import time # To measure the time of the loading of the files
import sys # To get a class from a string
import ast # To convert string to list

import uuid
from datetime import timedelta, datetime

import pandas as pd
import geopandas as gpd
import mesa
from shapely.geometry import Point
import osmnx as ox
from scipy.stats import skewnorm

import datetime as dt  # To keep track of the time of the simulation

from src.agent.worker import Worker
from src.agent.criminal import Criminal
from src.agent.police_agent import PoliceAgent
from src.space.city import City

directory = os.path.dirname(__file__)
for i in range(2):
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory

# TODO: fix the generation of the resting time for the workers on the first day
# TODO : check that the resident implementation of resting time (initizialization) does not create problems

class StreetCrime(mesa.Model):
    def __init__(
        self,
        p_criminals = 0.1,
        files= {
            "roads_file": directory + r"\data\processed\roads.graphml",
            "neighborhoods_file": directory + r"\data\processed\neighborhoods.gpkg",
            "buildings_file": directory + r"\data\processed\buildings.shp"},
        model_params = {'crs': "epsg:7791",
                    'len_step': 10,  # minutes
                    'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
                    'num_movers': 40,
                    'day_act_start': 8,
                    'day_act_end': 19,
                    'p_agents' : {'PoliceAgent' : 0.1}
                    },
        agent_params = {'Mover':    {"mean_work_start_time": 8,
                                "sd_work_start_time": 2,
                                "mean_work_end_time": 18,
                                "sd_work_end_time": 2,
                                "mean_self_defence": 0.5,
                                "sd_self_defence": 0.17,
                                "p_information": 0.5},
                    'InformedMover': {"p_information" : 1},
                    'Resident': {"mean_resting_start_time": 21,
                                "sd_resting_start_time": 2,
                                "mean_resting_end_time": 7.5,
                                "sd_resting_end_time": 0.83,},
                    'Worker':   {"mean_work_start_time": 8,
                                "sd_work_start_time": 2,
                                "mean_work_end_time": 18,
                                "sd_work_end_time": 2,
                                "mean_self_defence": 0.5,
                                "sd_self_defence": 0.17,
                                "p_information": 0.5},
                    'Criminal': {"mean_crime_motivation": 0.5,
                                "sd_crime_motivation": 0.17,
                                "opportunity_awareness": 300,
                                "crowd_effect": 0.01,
                                "p_information": 1},
                    'PoliceAgent': {"p_information": 1}}) -> None:
        super().__init__()
        self.running = True
        self.model_params = model_params
        self.agent_params = agent_params
        self.model_params['p_agents']['Criminal'] = p_criminals
        self.model_params['p_agents']['Worker'] = 1 - p_criminals - model_params['p_agents']['PoliceAgent']
        city = self._load_files(files)
        self.space = City(crs=model_params['crs'],
                          model=self,
                          road_network=city['road_network'], 
                          neighborhoods=city['neighborhoods'], 
                          buildings=city['buildings'])
        self.data = {
            'step_counter': 0,
            'crimes': gpd.GeoDataFrame(columns=['step', 'datetime', 'geometry', 'neighborhood',
                                                'victim', 'criminal', 'witnesses', 'successful']),
            'info_neighborhoods' : self.space.neighborhoods,
            'datetime': model_params['start_datetime'],
        }
        self.schedule = mesa.time.RandomActivation(self)
        self.create_movers()
        self.datacollector = mesa.DataCollector(
           model_reporters = self.data)

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
        city['road_network'] = ox.io.load_graphml(files["roads_file"])
        # Loading roads: --- 7.504880905151367 seconds ---
        #print("Loading roads: " + "--- %s seconds ---" %
        #      (time.time() - start_time))

        # Neighborhoods
        neighborhoods = gpd.read_file(files["neighborhoods_file"])
        neighborhoods.set_index('id', inplace=True)
        current_date = str(self.model_params['start_datetime'].date())
        neighborhoods = neighborhoods.assign(**{
            current_date + '_visits': 1,
            current_date + '_crimes': 1,
            current_date + '_police': 1,
            "run_visits" : 1,
            "run_crimes" : 1,
            "run_police" : 1,
            'city_income_distribution' : skewnorm(a = neighborhoods['city_ae'].iloc[0], 
                                                  loc = neighborhoods['city_loce'].iloc[0], 
                                                  scale =  neighborhoods['city_scalee'].iloc[0]),
        })
        neighborhoods.drop(['cap'], axis='columns', inplace=True) 
        city['neighborhoods'] = neighborhoods

        # Buildings
        start_time = time.time()
        buildings = gpd.read_file(files["buildings_file"])
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
        #Faster than list comprehension and map
        """Creates the movers of the model and adds them to the schedule and the space
        """
        start_time = time.time()
        movers = []
        for mover_type in self.model_params['p_agents']:
            class_type = getattr(sys.modules[__name__], mover_type)
            for _ in range(int(self.model_params['p_agents'][mover_type]*self.model_params['num_movers'])):
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
            timedelta(minutes=self.model_params['len_step'])
        # Only at midnight of each day after the first one, update information on crimes and visits of the previous day
        if (self.data['datetime'].day > 1 and self.data['datetime'].hour == 0 and self.data['datetime'].minute == 0):
            self.space.update_information()
        # Randomly compute step for each agent
        self.schedule.step()

Milan = StreetCrime()

for _ in range(500):
    Milan.step()
    