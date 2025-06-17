import os
import unittest

import geopandas as gpd
import networkx as nx
import osmnx as ox

from streetcrime.space.city import City


class TestCity(unittest.TestCase):
    def setUp(self) -> None:
        self.city = City(crs="epsg:32632", city_name="Duomo, Milan, Italy")

    """
        # Test with invalid crs
        out_file = "test_roads.graphml"
        city_name = "Milan, "
        crs = "invalid_crs"
        tolerance = 10
        traffic_factor = 2
        with self.assertRaises(NameError):
            obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)

        # Test with invalid tolerance
        out_file = "test_roads.graphml"
        city_name = "Milan, "
        crs = "EPSG:4326"
        tolerance = -10
        traffic_factor = 2
        with self.assertRaises(ValueError):
            obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)

        # Test with invalid city_name extension
        out_file = "test_roads.txt"
        city_name = "Milan, "
        crs = "EPSG:4326"
        tolerance = 10
        traffic_factor = 2
        with self.assertRaises(ValueError):
            obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)"""

    def test_obtaining_roads(self):
        # Test with .graphml file format
        out_file = "test_roads.graphml"
        city_name = "Duomo, Milan, Italy"
        crs = "epsg:4326"
        tolerance = 10
        traffic_factor = 2
        self.city.load_data(tolerance, traffic_factor, out_file)
        self.assertTrue(os.path.isfile(out_file))
        roads = ox.load_graphml(out_file)
        self.assertIsInstance(roads, nx.MultiDiGraph)
        os.remove(out_file)

        # Test with .gpkg file format
        out_file = "test_roads.gpkg"
        self.city.load_data(tolerance, traffic_factor, out_file)
        self.assertTrue(os.path.isfile(out_file))
        roads_nodes = gpd.read_file(out_file, layer="nodes").set_index("osmid")
        roads_edges = gpd.read_file(out_file, layer="edges").set_index(["u", "v", "key"])
        roads = ox.graph_from_gdfs(roads_nodes, roads_edges)
        self.assertIsInstance(roads, nx.MultiDiGraph)
        os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
