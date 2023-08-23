import pickle
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import momepy
import pyproj
import networkx as nx
import mesa
from sklearn.neighbors import KDTree
import os.path
import time

from src.space.utils import segmented


current_directory = os.path.dirname(__file__)
for _ in range(2):
    parent_directory = os.path.split(current_directory)[0]

    #TODO: change shortest path with sklearn.utils.graph_shortest_path.graph_shortest_path() and time

    def __init__(self, lines: gpd.GeoSeries):
        """The LineStrings are segmented to create segments going from node to node, and converted to nx

        Arguments:
            lines (gpd.GeoSeries[shapely.LineString]) -- The 'geometry' LineStrings segments series of the road network.
        """
        lines = gpd.GeoDataFrame(geometry=lines)
        G = momepy.gdf_to_nx(lines, length="length")
        for c in nx.connected_components(G) :
            self.sub_graphs = G.subgraph(c) 
        print(nx.connected_components(self.nx_graph))
        print(max(nx.connected_components(self.nx_graph)))

        self._kd_tree = KDTree(self.nx_graph.nodes)
        #TODO: select.path_cache_result see
        self._path_cache_result = os.path.join(current_directory, "outputs\_path_cache_result.pkl")
        try:
            with open(self._path_cache_result, "rb") as cached_result: #"rb" = read binary", "with" allows to open and close after execution 
                self._path_select_cache = pickle.load(cached_result)
        except (FileNotFoundError, EOFError):
            self._path_select_cache = dict()





