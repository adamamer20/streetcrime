import random

import mesa
import mesa_geo as mg
import pyproj

from shapely.geometry import Polygon
import geopandas as gpd
from simpledbf import Dbf5
import pandas as pd

WS = r'C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\\'

#Importing GIS files
def GIS_files():
    streets = gpd.read_file(WS+r'DBT_2020\DBT 2020 - SHAPE\AR_STR.shp')
    buildings = gpd.read_file(WS+r'DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp')
    buildings_category = Dbf5(WS+r'DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf')
    buildings_category = buildings_category.to_dataframe()
    buildings = buildings.merge(buildings_category, on = 'CLASSREF')
    return streets, buildings

class Building(mg.GeoAgent):
    unique_id: int  # an ID that represents the building
    model: mesa.Model
    geometry: Polygon
    crs: pyproj.CRS
    centroid: mesa.space.FloatCoordinate
    #name: str
    function: float  # 1.0 for work, 2.0 for home, 0.0 for neither
    entrance_pos: mesa.space.FloatCoordinate  # nearest vertex on road

    def __init__(self, unique_id, function, model, geometry, crs) -> None:
        super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)
        self.entrance = None
        self.function = None
        #self.name = str(uuid.uuid4()) #perchè non usare unique_id come nome?
        #self.function = randrange(3) #non voglio che sia generato randomicamente ma in base al tipo

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(unique_id={self.unique_id}, name={self.name}, function={self.function}, "
            f"centroid={self.centroid})"
        )
    
    def __eq__(self, other):
        if isinstance(other, Building):
            return self.unique_id == other.unique_id
        return False

class RoadNetwork:
    _nx_graph: nx.Graph
    _kd_tree: KDTree
    _crs: pyproj.CRS

    def __init__(self, lines: gpd.GeoSeries):
        segmented_lines = gpd.GeoDataFrame(geometry=segmented(lines))
        G = momepy.gdf_to_nx(segmented_lines, approach="primal", length="length")
        self.nx_graph = G.subgraph(max(nx.connected_components(G), key=len))
        self.crs = lines.crs

    @property
    def nx_graph(self) -> nx.Graph:
        return self._nx_graph

    @nx_graph.setter
    def nx_graph(self, nx_graph) -> None:
        self._nx_graph = nx_graph
        self._kd_tree = KDTree(nx_graph.nodes)

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @crs.setter
    def crs(self, crs) -> None:
        self._crs = crs

    def get_nearest_node(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> mesa.space.FloatCoordinate:
        node_index = self._kd_tree.query([float_pos], k=1, return_distance=False)
        node_pos = self._kd_tree.get_arrays()[0][node_index[0, 0]]
        return tuple(node_pos)

    def get_shortest_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> List[mesa.space.FloatCoordinate]:
        from_node_pos = self.get_nearest_node(source)
        to_node_pos = self.get_nearest_node(target)
        # return nx.shortest_path(self.nx_graph, from_node_pos, to_node_pos, method="dijkstra", weight="length")
        return nx.astar_path(self.nx_graph, from_node_pos, to_node_pos, weight="length")



class Milan(mg.GeoSpace):
    def __init__(self, crs, streets, buildings):
        super().__init__(crs="epsg:6707")
        self.streets = None
        self.buildings = None

    def load_datasets(self):
        self.streets = gpd.read_file(WS+r'DBT_2020\DBT 2020 - SHAPE\AR_STR.shp')
        buildings = gpd.read_file(WS+r'DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp')
        buildings_category = Dbf5(WS+r'DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf')
        buildings_category = buildings_category.to_dataframe()
        self.buildings = buildings.merge(buildings_category, on = 'CLASSREF')

        







class Resident(mesa_geo.GeoAgent):

    def __init__(self, unique_id, model, geometry, crs, p_police):
        super().__init__(unique_id, model, geometry, crs)
        self.crime_motivation = random.random()
        self.self_defence = random.random()
        self.policeman = random.random() <= p_police


class MilanStreetCrime(mesa.Model):

    def __init__(self, N):
        self.space = mg.GeoSpace()

        self.num_agents = N
        for i in range(self.num_agents):
            a = Resident(i, self)

def main():
    streets, buildings = GIS_files()


if __name__ == "__main__":
    main()




