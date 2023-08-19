import time
import geopandas as gpd

import mesa 
import mesa_geo as mg
from shapely.geometry import Point

from src.agent.mover import Mover
from src.agent.resident import Resident
from src.agent.worker import Worker
from src.agent.criminal import Criminal
from src.space.road_network import RoadNetwork


class City(mg.GeoSpace):
    """
    The City class is the GeoSpace in which agents move.
    It is used to store the road network and the buildings and neighborhoods dataframes.
    ...
    
    Args:
        crs(str)
            The crs of the city
        roads(RoadNetwork)
            The road network of the city (see thesis\src\space\road_network.py)
        neighborhoods_df(gpd.GeoDataFrame)
            The neighborhoods dataframe of the city
        buildings_df(str)
            The buildings dataframe of the city
    Methods:
        get_random_building(resident : Resident, function : str) -> int
            Returns a random building id based on the function passed as argument
        find_neighborhood_by_position(position : Point) -> int
            Returns the neighborhood id of the neighborhood in which the position passed as argument is contained
        distance_from_buildings(position : Point) -> gpd.GeoSeries
            Returns the distance from the position passed as argument to all the buildings in the city
        update_information() -> None
            Updates 'yesterday_crimes' and 'yesterday_visits' columns of the buildings dataframe with data
            collected in 'today_crimes' and 'today_visits' columns of the neighborhoods dataframe
    """

    def __init__(self, crs: str, 
                 roads_df: gpd.GeoDataFrame, neighborhoods_df:gpd.GeoDataFrame, buildings_df:gpd.GeoDataFrame) -> None:
        #TODO: specify columns of the dataframes
        """
        Args:
            crs(str)
                The crs of the city
            model(mesa.Model)
                The model of the simulation where the GeoSpace is used
            roads_df(gpd.GeoDataFrame)
                The roads dataframe of the city
            neighborhoods_df(gpd.GeoDataFrame)
                The neighborhoods dataframe of the city
            buildings_df(str)
                The buildings dataframe of the city
        """
        super().__init__(crs=crs)
        self.roads = RoadNetwork(roads_df["geometry"])
        self.neighborhoods_df = neighborhoods_df
        self.buildings_df = buildings_df

    def get_random_building(self, resident: Resident = None, function: str = None) -> int:
        """This method act on the City.buildings_df to obtain a random building id based on the function and the Resident that requested it.
        Args:
            resident (Resident, optional): The resident that is looking for a building, by default None.
            function (str, optional): The function of the building that the resident is looking for, by default None. Can be "home", "work", "day_act" or "night_act".  
        
        Returns:
            int: The id of the building
        
        """
        match function:
            case "home": 
                match resident:
                    case Criminal():
                        #Weights = proportion of population in each neighborhood * (1/income)
                        weights = self.neighborhoods_df['prop'] * (1/self.neighborhoods_df['income'])
                    case _:
                        #Weights = proportion of population in each neighborhood
                        weights = self.neighborhoods_df['prop']
                _neighborhood_df = self.neighborhoods_df.sample(n=1, weights=weights)
                #_building_df = self.buildings_df[(self.buildings_df[function] == True) & (self.buildings_df['neighborho'] == _neighborhood_df.index[0])].sample(n = 1)
                # TODO: RIGHT ONE BUT USE IT ONLY WITH ALL BUILDINGS, use only building df at the end of the function to avoid repetition
                _building_df = self.buildings_df[(self.buildings_df[function] == True)].sample(n=1)
            case "day_act", "night_act":
                match resident:
                    case Worker():
                        #The worker chooses based on the distance from him and the known number of yesterday crimes in the neighborhood 
                        if resident.model.datetime.day > 1:
                            resident.criminal_neighborhoods.reset_index(
                                names='neighborho', inplace=True)
                            _buildings_df = self.buildings_df.drop(
                                ['yesterday_crimes'], axis='columns')
                            _buildings_df = _buildings_df.merge(
                                resident.criminal_neighborhoods, on='neighborho').set_axis(self.buildings_df.index)
                            weights = (1/self.distance_from_buildings(resident.geometry)) * (1/_buildings_df['yesterday_crimes'])
                        elif resident.model.datetime.day == 1:
                            weights = 1/self.distance_from_buildings(resident.geometry)
                    case Criminal():
                        #The criminal chooses based on the distance from him and the known number of yesterday visits in the neighborhood
                        if resident.model.datetime.day > 1:
                            weights = (1/self.distance_from_buildings(resident.geometry)) * (1/self.buildings_df['yesterday_visits'])
                        elif resident.model.datetime.day == 1:
                            weights = 1/self.distance_from_buildings(resident.geometry)
                    case _:
                        weights = 1/self.distance_from_buildings(resident.geometry)
                _building_df = self.buildings_df[(self.buildings_df[function] is True)].sample(
                    n=1, weights=weights)
            case _:
                _building_df = self.buildings_df.sample(n=1)
        return _building_df.index[0]

    def find_neighborhood_by_position(self, position: Point) -> int:
        """Find the neighborhood in which the position passed as argument is contained

        Args:
            position (Point): The position to find the containing neighborhood

        Returns:
            int: The id of the neighborhood
        """
        return self.neighborhoods_df[self.neighborhoods_df['geometry'].contains(position)].index[0]

    def distance_from_buildings(self, position: Point) -> gpd.GeoSeries:
        """Find the distance from the position passed as argument to all the buildings in the city

        Args:
            position (Point): The position to find the distance from

        Returns:
            gpd.GeoSeries: The distances from the position to all the buildings in the city, ordered as self.buildings_df
        """
        return self.buildings_df['geometry'].distance(position)

    def update_information(self) -> None:
        """Updates 'yesterday_crimes' and 'yesterday_visits' columns of the self.buildings_df with data
        collected in 'today_crimes' and 'today_visits' columns of the self.neighborhoods_df dataframe.
        This method is only initiated if the model.worker_params.information = 1 (Perfect Information).
        """
        _info_neighborhoods = self.neighborhoods_df[[
            'today_visits', 'today_crimes']]
        _info_neighborhoods.reset_index(inplace=True)
        _info_neighborhoods.rename(columns={'ID_NIL': 'neighborho',
                                            'today_visits': 'yesterday_visits',
                                            'today_crimes': 'yesterday_crimes'}, inplace=True)
        self.buildings_df.drop(
            ['yesterday_visits', 'yesterday_crimes'], inplace=True, axis='columns')
        self.buildings_df = self.buildings_df.merge(
            _info_neighborhoods, on='neighborho').set_axis(self.buildings_df.index)
        # Initializing at 1,1 avoid moltiplication by 0 when calculating weights in self.model.space.get_random_building
        self.neighborhoods_df = self.neighborhoods_df.assign(
            today_visits=1, today_crimes=1)
