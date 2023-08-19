import pickle
from typing import Dict, List, Tuple, Optional

from sklearnex import patch_sklearn

import geopandas as gpd
import momepy
import pyproj
import networkx as nx
import mesa
from sklearn.neighbors import KDTree
import os.path
import time

from src.space.utils import segmented

patch_sklearn()

current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]
    current_directory = parent_directory

#TODO: Why properties for the nx_graph and crs? Why not just use the nx_graph and crs directly?
class RoadNetwork:
    """The RoadNetwork class is used to store the road network of the city as a NetworkX graph. 
    The code is a modified version of project-mesa/mesa-examples/gis/agents-and-networks/src/space/road_network.py.
    
    Arguments:
        lines (gpd.GeoSeries[shapely.LineString]) -- The 'geometry' series of the road network.  They are LineStrings.

    Returns:
        _description_
    """
    _nx_graph: nx.Graph
    _kd_tree: KDTree 
    _crs: pyproj.CRS
    _path_select_cache: Dict[
        Tuple[mesa.space.FloatCoordinate, mesa.space.FloatCoordinate],
        List[mesa.space.FloatCoordinate],
    ]

    def __init__(self, lines: gpd.GeoSeries):
        """The LineStrings are segmented to create segments going from node to node, and converted to nx

        Arguments:
            lines (gpd.GeoSeries[shapely.LineString]) -- The 'geometry' series of the road network.  They are LineStrings.
        """
        segmented_lines = gpd.GeoDataFrame(geometry=segmented(lines))
        G = momepy.gdf_to_nx(segmented_lines, length="length")
        print(nx.connected_components(G))
        print(max(nx.connected_components(G)))
        self.nx_graph = G
        #self.nx_graph = G.subgraph(max(nx.connected_components(G), key=len))
        self.crs = lines.crs
        self._path_cache_result = os.path.join(current_directory, "outputs\_path_cache_result.pkl")
        try:
            with open(self._path_cache_result, "rb") as cached_result: #"rb" = read binary", "with" allows to open and close after execution 
                self._path_select_cache = pickle.load(cached_result)
        except (FileNotFoundError, EOFError):
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
        return nx.astar_path(self.nx_graph, from_node_pos, to_node_pos, weight="length") #NetworkX is a pure python library so it's slow. find something faster

    def cache_path(
        self,
        source: mesa.space.FloatCoordinate,
        target: mesa.space.FloatCoordinate,
        path: List[mesa.space.FloatCoordinate],
    ) -> None:
        from_node_pos = self.get_nearest_node(source)
        to_node_pos = self.get_nearest_node(target)
        print(f"caching path... current number of cached paths: {len(self._path_select_cache)}")
        self._path_select_cache[(from_node_pos, to_node_pos)] = path
        self._path_select_cache[(to_node_pos, from_node_pos)] = list(reversed(path))
        with open(self._path_cache_result, "wb") as cached_result:
            pickle.dump(self._path_select_cache, cached_result)

    def get_cached_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> Optional[List[mesa.space.FloatCoordinate]]:
        from_node_pos = self.get_nearest_node(source)
        to_node_pos = self.get_nearest_node(target)
        return self._path_select_cache.get((from_node_pos, to_node_pos), None)

