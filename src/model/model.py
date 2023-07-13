import uuid
from functools import partial
from fiona.errors import DriverError

import pandas as pd
import geopandas as gpd
import mesa
import mesa_geo as mg
from shapely.geometry import Point


from src.agent.resident import Resident
from src.agent.geo_agents import Road
from src.space.building import Building
from src.space.city import City
from src.space.road_network import RoadNetwork
from src.space.building import data_processing


def get_time(model) -> pd.Timedelta:
    return pd.Timedelta(days=model.day, hours=model.hour, minutes=model.minute)


def get_num_residents_by_status(model, status: str) -> int:
    residents = [
        resident for resident in model.schedule.agents if resident.status == status
    ]
    return len(residents)


class StreetCrime(mesa.Model):
    running: bool
    schedule: mesa.time.RandomActivation
    current_id: int
    space: City
    roads: RoadNetwork
    world_size: gpd.geodataframe.GeoDataFrame
    got_to_destination: int  # count the total number of arrivals
    num_residents: int
    day: int
    hour: int
    minute: int
    datacollector: mesa.DataCollector

    def __init__(
        self,
        buildings_file: str,
        buildings_fun_file: str,
        roads_file: str,
        crs: str = "epsg:7791",
        num_residents: int = 10,
        resident_speed: float = 1.0,
        show_roads : bool = False
    ) -> None:
        super().__init__()
        self.crs = crs
        self.schedule = mesa.time.RandomActivation(self)
        self.show_roads = show_roads
        self.space = City(crs=crs)
        self.num_residents = num_residents
        self._load_buildings_from_file(buildings_file, buildings_fun_file)
        self._load_road_vertices_from_file(roads_file)
        self._set_building_entrance()
        self.got_to_destination = 0
        self._create_residents()
        self.day = 0
        self.hour = 5
        self.minute = 55

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "time": get_time,
                "status_home": partial(get_num_residents_by_status, status="home"),
                "status_work": partial(get_num_residents_by_status, status="work"),
                "status_traveling": partial(
                    get_num_residents_by_status, status="transport"
                )
            }
        )
        self.datacollector.collect(self)

    #Loading files
    def _load_buildings_from_file(
        self, buildings_file: str, buildings_fun_file: str
    ) -> None:
        try:
            buildings_df = gpd.read_file(r'C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\processed\buildings.shp')
            buildings_df[['home', 'day_act','night_act']] = buildings_df[['home', 'day_act','night_act']].astype(bool)
        except DriverError:
            buildings_df = data_processing(buildings_file, buildings_fun_file)
        buildings_df["position"] = [
            (x, y) for x, y in zip(buildings_df.centroid.x, buildings_df.centroid.y)
        ]
        self.space.add_buildings(buildings_df)

    def _load_road_vertices_from_file(
        self, roads_file: str
    ) -> None:
        roads_df = gpd.read_file(roads_file)
        self.roads = RoadNetwork(lines=roads_df["geometry"])
        roads_creator = mg.AgentCreator(Road, model=self)
        roads = roads_creator.from_GeoDataFrame(roads_df)
        self.space.add_agents(roads)

    def _set_building_entrance(self) -> None:
        self.space.buildings_df['entrance_pos'] = [
                    self.roads.get_nearest_node(x) 
                    for x in self.space.buildings_df['position']]

    def _create_residents(self) -> None:
        for _ in range(self.num_residents):
            random_home = self.space.get_random_home()
            random_work = self.space.get_random_work()
            resident = Resident(
                unique_id=uuid.uuid4().int,
                model=self,
                geometry=Point(random_home["position"].item()),
                crs=self.space.crs,
            )
            resident.set_home(random_home)
            resident.set_work(random_work)
            resident.status = "home"
            self.space.add_resident(resident)
            self.schedule.add(resident)

    def step(self) -> None:
        self.__update_clock()
        self.schedule.step()
        self.datacollector.collect(self)

    def __update_clock(self) -> None:
        self.minute += 30
        if self.minute == 60:
            if self.hour == 23:
                self.hour = 0
                self.day += 1
            else:
                self.hour += 1
            self.minute = 0

#Milan = StreetCrime(buildings_file = r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp",
#                    buildings_fun_file = r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf", 
#                    roads_file = r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\processed\EL_STR.shp")
