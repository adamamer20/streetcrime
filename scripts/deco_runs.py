import numpy as np
import os.path
import time
import datetime as dt
from src.model import StreetCrime
import uuid 
from deco import concurrent, synchronized
import geopandas as gpd
import osmnx as ox
from scipy.stats import skewnorm

@concurrent(4) # We add this for the concurrent function
def run(city, 
        model_params = {
            'crs': "epsg:7791",
            'days': 5,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            'num_movers': 500,
            'day_act_start': 8,
            'day_act_end': 19,
            'p_agents' : {'PoliceAgent' : 0.1,
                            'Pickpocket' : 0.1,
                            'Robber' : 0.05},
            },
        agents_params = {
            'Resident': {"rmean_resting_start_time": 21,
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
            'Criminal': {"opportunity_awareness" : 150,
                        "crowd_effect": 0.01,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"p_information": 1}}):
    run_id = uuid.uuid4()
    Milan = StreetCrime(
        run_id = run_id,
        city = city,
        model_params = model_params,
        agents_params = agents_params)
    start_time = time.time()
    for i in range(Milan.model_params['n_steps']):
        step_time = time.time()
        Milan.step()
        print(f"{i} step took: " + "--- %s seconds ---" % (time.time() - step_time))
    print(f"{Milan.model_params['n_steps']} steps for {Milan.model_params['num_movers']} movers took: " + "--- %s seconds ---" % (time.time() - start_time))
    output = Milan.get_data()
    output.to_pickle(f"outputs/runs/run_{run_id}.pkl")
    return output

@synchronized
def multiple_runs():
    directory = os.path.dirname(__file__)
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory

    os.chdir(directory)

    city = load_files()
    city['directory'] = directory
    
    #sensitivty_analysis(city)
    for _ in range(2):
        run(city = city, 
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
            'Criminal': {"opportunity_awareness" : 150,
                        "crowd_effect": 0.01,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"act_decision_rule": ""}})
    
    for _ in range(2):
        run(city = city, 
            model_params = {
                'crs': "epsg:7791",
                'days': 14,
                'len_step': 15,  # minutes
                'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
                'num_movers': 600,
                'day_act_start': 8,
                'day_act_end': 19,
                'p_agents' : {'PoliceAgent' : 0.02,
                                'Pickpocket' : 0.04,
                                'Robber' : 0.04},
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
            'Criminal': {"opportunity_awareness" : 150,
                        "crowd_effect": 0.01,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"act_decision_rule": ""}})
     
#def long_run(city):
        
            
def sensitivity_analysis(city):
    for p_info_worker in np.arange(0.4, 1, 0.3):
        run(
            city = city, 
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
                        "p_information": p_info_worker},
            'Criminal': {"opportunity_awareness" : 150,
                        "crowd_effect": 0.01,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"p_information": 1}},
            )    

    for crowd_effect in np.arange(0.01, 0.1, 0.03):
        run(
            city = city,
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
            'Criminal': {"opportunity_awareness" : 150,
                        "crowd_effect": crowd_effect,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"p_information": 1}})

    for opportunity_awareness in np.arange(100, 300, 100):
        run(
            city = city,
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
            'Criminal': {"opportunity_awareness" : opportunity_awareness,
                        "crowd_effect": 0.01,
                        "p_information": 1},
            'Pickpocket' : {"act_decision_rule": "1/np.log10(distance) * yesterday_visits * run_visits * mean_income * (1/yesterday_police) * (1/run_police)"},
            'Robber' : {"act_decision_rule" : "1/np.log10(distance) * (1/yesterday_visits) * (1/run_visits) * mean_income * (1/yesterday_police) * (1/run_police)"},
            'PoliceAgent': {"p_information": 1}})
        
    for p_pickpocket in np.arange(0.02, 0.15, 0.03):
        run(
            city = city,
            model_params = {
            'crs': "epsg:7791",
            'days': 31,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            'num_movers': 1000,
            'day_act_start': 8,
            'day_act_end': 19,
            'p_agents' : {'PoliceAgent' : 0.02,
                            'Pickpocket' : p_pickpocket,
                            'Robber' : 0.02},
            })

    for p_robber in np.arange(0.02, 0.15, 0.03):
        run(
            city = city,
            model_params = {
            'crs': "epsg:7791",
            'days': 3,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            'num_movers': 1000,
            'day_act_start': 8,
            'day_act_end': 19,
            'p_agents' : {'PoliceAgent' : 0.02,
                            'Pickpocket' : 0.04,
                            'Robber' : p_robber},
            })

    for p_police in np.arange(0.01, 0.10, 0.02):
        run(
            city = city,
            model_params = {
            'crs': "epsg:7791",
            'days': 3,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            'num_movers': 1000,
            'day_act_start': 8,
            'day_act_end': 19,
            'p_agents' : {'PoliceAgent' : p_police,
                            'Pickpocket' : 0.04,
                            'Robber' : 0.02},
            })
          
def load_files(start_datetime = dt.datetime(2020, 1, 1, 5, 30),
               files = {
                "roads_file":  "data/processed/roads.gpkg",
                "public_transport_file": "data/processed/public_transport.gpkg",
                "neighborhoods_file": "data/processed/neighborhoods.gpkg",
                "buildings_file": "data/processed/buildings.shp"},
               ):
    """Load the files of the city and return a dictionary of GeoDataFrames or nx.MultiDiGraphs with the loaded files

    Parameters
    ----------
    files : dict[str, str]
        The files to load. The keys are the names of the files and the values are the paths to the files

    Returns
    -------
    dict[str, gpd.GeoDataFrame | nx.MultiDiGraph]
        A dictionary of GeoDataFrames or nx.MultiDiGraphs with the loaded files. 
            - roads_nodes: The nodes of the roads of the city
            - roads_edges: The edges of the roads of the city
            - roads: The MultiDiGraph of the roads of the city
            - public_transport_nodes: The nodes of the public transport of the city
            - public_transport_edges: The edges of the public transport of the city
            - public_transport: The MultiDiGraph of the public transport of the city
            - neighborhoods: The neighborhoods of the city
            - buildings: The buildings of the city
    """
    start_time = time.time()
    city = {}

    # Roads & public transport
    city['roads_nodes'] = gpd.read_file(files["roads_file"], layer='nodes').set_index('osmid')
    city['roads_edges'] = gpd.read_file(files["roads_file"], layer='edges').set_index(['u', 'v', 'key'])
    city['roads'] = ox.graph_from_gdfs(city['roads_nodes'], city['roads_edges'])
    city['public_transport_nodes'] = gpd.read_file(files["public_transport_file"], layer='nodes').set_index('osmid')
    city['public_transport_edges'] = gpd.read_file(files["public_transport_file"], layer='edges').set_index(['u', 'v', 'key'])
    city['public_transport'] = ox.graph_from_gdfs(city['public_transport_nodes'], city['public_transport_edges'])

    # Neighborhoods
    neighborhoods = gpd.read_file(files["neighborhoods_file"]).set_index('id')
    start_date = str(start_datetime.date())
    # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
    neighborhoods = neighborhoods.assign(**{
        start_date + '_visits': 1,
        start_date + '_crimes': 1,
        start_date + '_police': 1,
        "run_visits" : 1,
        "run_crimes" : 1,
        "run_police" : 1,
        'city_income_distribution' : skewnorm(a = neighborhoods['city_ae'].iloc[0], 
                                                loc = neighborhoods['city_loce'].iloc[0], 
                                                scale =  neighborhoods['city_scalee'].iloc[0]),
    })
    neighborhoods.rename(columns={'mean income': 'mean_income'}, inplace=True)
    neighborhoods.drop(['cap'], axis='columns', inplace=True) 
    city['neighborhoods'] = neighborhoods

    # Buildings
    buildings = gpd.read_file(files["buildings_file"]).set_index('id')
    buildings[['home', 'day_act', 'night_act']] = buildings[[
        'home', 'day_act', 'night_act']].astype(bool)
    buildings.rename(columns={'neighborho': 'neighborhood'}, inplace=True)
    buildings['neighborhood'] = buildings['neighborhood'].astype(int)
    buildings = buildings.merge(neighborhoods['mean_income'], left_on = "neighborhood", right_index = True)
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
    
    print("Loaded files: " + "--- %s seconds ---" %
            (time.time() - start_time))
    
    return city

if __name__ == '__main__':
    multiple_runs()