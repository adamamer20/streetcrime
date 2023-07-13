import random
from collections import defaultdict
from typing import Dict, Tuple, Optional, DefaultDict, Set #Type hinting
import geopandas as gpd

import mesa
import mesa_geo as mg
from shapely.geometry import Point

from src.agent.resident import Resident
from src.space.building import Building


class City(mg.GeoSpace):
    buildings_df: gpd.GeoDataFrame
    _buildings: Dict[int, Building]
    _residents_pos_map: DefaultDict[mesa.space.FloatCoordinate, Set[Resident]]
    _resident_id_map: Dict[int, Resident]

    def __init__(self, crs: str) -> None:
        super().__init__(crs=crs)
        self.buildings = tuple()
        self._residents_pos_map = defaultdict(set)
        self._resident_id_map = dict()
        self.buildings_df = gpd.GeoDataFrame()

    def add_buildings(self, buildings_df) -> None:
        self.buildings_df = buildings_df
        buildings_creator = mg.AgentCreator(Building, model=self)
        buildings = buildings_creator.from_GeoDataFrame(buildings_df)
        super().add_agents(buildings)

    def add_resident(self, agent: Resident) -> None:
        super().add_agents([agent])
        self._residents_pos_map[(agent.geometry.x, agent.geometry.y)].add(agent)
        self._resident_id_map[agent.unique_id] = agent

    def get_random_home(self) -> Building:
        #Select a random building from the buildings_df that has the attribute 'is_home' set to True
        return self.buildings_df[(self.buildings_df['home'] == True)].sample(n = 1)
         
    def get_random_work(self) -> Building:
        return self.buildings_df.sample(n = 1)
    
    def get_random_day_activity(self) -> Building: #Non dovrebbe essere casuale
        return self.buildings_df.loc[(self.buildings_df['day_act'] == True)].sample(n = 1)
    
    def get_random_night_activity(self) -> Building: #Non dovrebbe essere casuale
        return self.buildings_df.loc[(self.buildings_df['night_act'] == True)].sample(n = 1)

    def get_residents_by_pos(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> Set[Resident]:
        return self._residents_pos_map[float_pos]

    def get_resident_by_id(self, resident_id: int) -> Resident:
        return self._resident_id_map[resident_id]

    def move_resident(
        self, resident: Resident, pos: mesa.space.FloatCoordinate
    ) -> None:
        self.__remove_resident(resident)
        resident.geometry = Point(pos)
        self.add_resident(resident)

    def __remove_resident(self, resident: Resident) -> None:
        super().remove_agent(resident)
        del self._resident_id_map[resident.unique_id]
        self._residents_pos_map[(resident.geometry.x, resident.geometry.y)].remove(
            resident
        )


