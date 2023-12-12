import numpy as np
import os.path
import time
import datetime as dt
from streetcrime.model import StreetCrime
import uuid 
import osmnx as ox
from scipy.stats import skewnorm
import geopandas as gpd

def run(): 
    directory = os.path.dirname(__file__)
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory

    range_opportunity_awareness = np.arange(100, 450, 50)
    range_p_pickpocket = np.arange(0.02, 0.20, 0.03)
    range_p_robber = np.arange(0.02, 0.20, 0.03)
    range_p_police = np.arange(0.01, 0.10, 0.02)
    range_p_info_worker = np.arange(0.4, 1, 0.1)
    range_crowd_effect = np.arange(0.01, 0.15, 0.02)

    for opportunity_awareness in range_opportunity_awareness:
        for crowd_effect in range_crowd_effect:
            for p_pickpocket in range_p_pickpocket:
                for p_robber in range_p_robber:
                    for p_police in range_p_police:
                        for p_info_worker in range_p_info_worker:
                            run_id = uuid.uuid4()
                            Milan = StreetCrime(
                                files= {
                                    "directory" : directory,
                                    "roads_file":  "data/processed/roads.gpkg",
                                    "public_transport_file": r"data\processed\public_transport.gpkg",
                                    "neighborhoods_file": r"data\processed\neighborhoods.gpkg",
                                    "buildings_file": r"data\processed\buildings.shp"},
                                model_params = {
                                    'crs': "epsg:7791",
                                    'days': 3,
                                    'len_step': 15,  # minutes
                                    'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
                                    'num_movers': 500,
                                    'day_act_start': 8,
                                    'day_act_end': 19,
                                    'p_agents' : {'PoliceAgent' : p_police,
                                                    'Pickpocket' : p_pickpocket,
                                                    'Robber' : p_robber},
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
                                                "p_information": p_info_worker},
                                    'Criminal': {"opportunity_awareness" : opportunity_awareness,
                                                "crowd_effect": crowd_effect,
                                                "p_information": 1},
                                    'PoliceAgent': {"p_information": 1}})
                            start_time = time.time()
                            for i in range(Milan.model_params['n_steps']):
                                step_time = time.time()
                                Milan.step()
                                print(f"{i} step took: " + "--- %s seconds ---" % (time.time() - step_time))
                            print(f"{Milan.model_params['n_steps']} steps for {Milan.model_params['num_movers']} movers took: " + "--- %s seconds ---" % (time.time() - start_time))
                            output = Milan.get_data(run_id)
                            output.to_pickle(os.path.join(parent_directory, f"outputs/runs/run_{run_id}.pkl"))

def load_files(start_datetime = dt.datetime(2020, 1, 1, 5, 30),
               files = {
                "roads_file":  "data/processed/roads.gpkg",
                "public_transport_file": r"data\processed\public_transport.gpkg",
                "neighborhoods_file": r"data\processed\neighborhoods.gpkg",
                "buildings_file": r"data\processed\buildings.shp"},
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

def run(run_id,
        city, 
        model_params = {
            'crs': "epsg:7791",
            'days': 3,
            'len_step': 15,  # minutes
            'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
            'num_movers': 5,
            'day_act_start': 8,
            'day_act_end': 19,
            'p_agents' : {'PoliceAgent' : 0.02,
                            'Pickpocket' : 0.04,
                            'Robber' : 0.02},
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
            'PoliceAgent': {"p_information": 1}}):
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
    return output

def main():
    directory = os.path.dirname(__file__)
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory

    os.chdir(directory)

    city = load_files()
    city['directory'] = directory
    
    outputs = {}
    run_id = uuid.uuid4()
    outputs[run_id] = run(
        run_id = run_id,
        city = city,
        model_params = {
        'crs': "epsg:7791",
        'days': 1,
        'len_step': 15,  # minutes
        'start_datetime': dt.datetime(2020, 1, 1, 5, 30),
        'num_movers': 5,
        'day_act_start': 8,
        'day_act_end': 19,
        'p_agents' : {'PoliceAgent' : 0.02,
                        'Pickpocket' : 0.04,
                        'Robber' : 0.02},
        })
    for run_id, output in outputs.items():
        output.to_pickle(f"outputs/runs/run_{run_id}.pkl")
        
if __name__ == "__main__":
    main()