import pickle
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import momepy
import pyproj
import networkx as nx
import mesa
from sklearn.neighbors import KDTree

from src.space.utils import segmented

#test da eliminare
import matplotlib.pyplot as plt

class RoadNetwork:
    unique_id: str
    model: mesa.Model
    crs: pyproj.CRS
    _nx_graph: nx.Graph
    _kd_tree: KDTree 
    _crs: pyproj.CRS
    _path_select_cache: Dict[
        Tuple[mesa.space.FloatCoordinate, mesa.space.FloatCoordinate],
        List[mesa.space.FloatCoordinate],
    ]

    def __init__(self, lines: gpd.GeoSeries):
        segmented_lines = gpd.GeoDataFrame(geometry=segmented(lines))
        G = momepy.gdf_to_nx(segmented_lines, approach="primal", length="length")
        self.nx_graph = G.subgraph(max(nx.connected_components(G), key=len))
        self.crs = lines.crs
        self._path_cache_result = r"outputs/_path_cache_result.pkl"
        try:
            with open(self._path_cache_result, "rb") as cached_result: #"rb" = read binary", "with" allows to open and close after execution 
                self._path_select_cache = pickle.load(cached_result)
        except FileNotFoundError:
            self._path_select_cache = dict()

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

    def cache_path(
        self,
        source: mesa.space.FloatCoordinate,
        target: mesa.space.FloatCoordinate,
        path: List[mesa.space.FloatCoordinate],
    ) -> None:
        # print(f"caching path... current number of cached paths: {len(self._path_select_cache)}")
        self._path_select_cache[(source, target)] = path
        self._path_select_cache[(target, source)] = list(reversed(path))
        with open(self._path_cache_result, "wb") as cached_result:
            pickle.dump(self._path_select_cache, cached_result)

    def get_cached_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> Optional[List[mesa.space.FloatCoordinate]]:
        return self._path_select_cache.get((source, target), None)

