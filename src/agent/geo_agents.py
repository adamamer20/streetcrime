import mesa
import mesa_geo as mg
from shapely.geometry import Point
import pyproj

class Road(mg.GeoAgent):
    unique_id: int
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)
