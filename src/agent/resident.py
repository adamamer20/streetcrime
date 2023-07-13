from typing import List #Type hinting
import math #Extract working hour

import pyproj #Type hinting for CRS
from src.space.building import Building #Type hinting for Building

import numpy as np #Generate working hours

import mesa
import mesa_geo as mg
from shapely.geometry import Point, LineString

from src.space.utils import redistribute_vertices, UnitTransformer
from src.space.road_network import RoadNetwork


class Resident(mg.GeoAgent):
    unique_id: int  # resident_id, used to link residents and nodes
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS
    origin: Building  # where he begins his trip
    destination: Building  # the destination he wants to arrive at
    path: List[
        mesa.space.FloatCoordinate
    ]  # a set containing nodes to visit in the shortest path
    home: Building
    work: Building
    work_start_time: List  # time to start going to work, [h, m]
    work_end_time: List  # time to leave work [h,m]
    _work_start_time: float
    _work_end_time: float
    status: str  # work, home, or transport
    SPEED: float

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)
        #Generate working hours
        _work_start_time = np.random.normal(9, 0.5)
        while _work_start_time < 6 or _work_start_time > 11: #Start of working time centered on 9 with 05 deviation with limit 6 and 11
            _work_start_time = np.random.normal(9, 0.5)
        self.work_start_time = [math.floor(_work_start_time), (_work_start_time % 1) * 60]
        _work_end_time = _work_start_time + np.random.choice([1, 1.5, 2])  # will work for 8 hours (with 1h, 1.5h or 2h break)
        self._work_end_time = [math.floor(_work_end_time), (_work_end_time % 1) * 60]
        self.step_in_path = 0

    def __repr__(self) -> str:
        return (
            f"Resident(unique_id={self.unique_id}, geometry={self.geometry}, status={self.status}"
        )

    def set_home(self, home: Building) -> None:
        self.home = home

    def set_work(self, work: Building) -> None:
        self.work = work

    def step(self) -> None:
        self._prepare_to_move()
        self._move()

    def _prepare_to_move(self) -> None:
        # start going to work
        if (
            self.status == "home"
            and self.model.hour == self.work_start_time[0] #il tempo in cui la persona parte deve essere sistemato
            ):
            self.origin = self.home
            self.model.space.move_resident(self, pos=self.origin.position)
            self.destination = self.work
            self._path_select()
            self.status = "transport"
        # start going home
        elif (
            self.status == "work"
            and self.model.hour == self.work_end_time[0]
            and self.model.minute == self.work_end_time[1]
        ):
            self.origin = self.work.position
            self.model.space.move_resident(self, pos=self.origin.position)
            self.destination = self.home.position
            self._path_select()
            self.status = "transport"

    def _move(self) -> None:
        if self.status == "transport":
            if self.step_in_path < len(self.path):
                next_position = self.path[self.step_in_path]
                self.model.space.move_resident(self, next_position)
                self.step_in_path += 1
            else:
                self.model.space.move_resident(self, self.destination.centroid)
                if self.destination == self.work:
                    self.status = "work"
                elif self.destination == self.home:
                    self.status = "home"
                self.model.got_to_destination += 1

    def advance(self) -> None:
        raise NotImplementedError

    def _path_select(self) -> None:
        self.step_in_path = 0
        if (
            cached_path := self.model.roads.get_cached_path(
                source=self.origin.entrance_pos, target=self.destination.entrance_pos
            )
        ) is not None:
            self.path = cached_path
        else:
            self.path = self.model.roads.get_shortest_path(
                source=self.origin.entrance_pos, target=self.destination.entrance_pos
            )
            self.model.roads.cache_path(
                source=self.origin.entrance_pos,
                target=self.destination.entrance_pos,
                path=self.path,
            )
        self._redistribute_path_vertices()

    def _redistribute_path_vertices(self) -> None:
        # if origin and destination share the same entrance, then self.path will contain only this entrance node,
        # and len(self.path) == 1. There is no need to redistribute path vertices.
        if len(self.path) > 1:
            unit_transformer = UnitTransformer(degree_crs=self.model.roads.crs)
            original_path = LineString([Point(p) for p in self.path])
            # from degree unit to meter
            path_in_meters = unit_transformer.degree2meter(original_path)
            redistributed_path_in_meters = redistribute_vertices(
                path_in_meters, self.SPEED
            )
            # meter back to degree
            redistributed_path_in_degree = unit_transformer.meter2degree(
                redistributed_path_in_meters
            )
            self.path = list(redistributed_path_in_degree.coords)

