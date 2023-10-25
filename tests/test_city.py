import unittest
import geopandas as gpd
import osmnx as ox
from src.space.city import City

class TestCity(unittest.TestCase):
    '''def test_obtaining_roads(self):
        # Test with .graphml file format
        out_file = "test_roads.graphml"
        city_name = "Milan, Italy"
        crs = "EPSG:4326"
        tolerance = 10
        traffic_factor = 2
        obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)
        self.assertTrue(os.path.isfile(out_file))
        roads = ox.load_graphml(out_file)
        self.assertIsInstance(roads, nx.MultiDiGraph)
        os.remove(out_file)

        # Test with .gpkg file format
        out_file = "test_roads.gpkg"
        obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)
        self.assertTrue(os.path.isfile(out_file))
        roads_nodes = gpd.read_file(out_file, layer = 'nodes').set_index('osmid')
        roads_edges = gpd.read_file(out_file, layer = 'edges').set_index(['u', 'v', 'key'])
        roads = ox.graph_from_gdfs(roads_nodes, roads_edges)
        self.assertIsInstance(roads, nx.MultiDiGraph)
        os.remove(out_file)

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
            obtaining_roads(out_file, city_name, crs, tolerance, traffic_factor)'''
    
    def test_city_init(self):
        City('epsg:7991',
             roads = 'tests/data/processed/roads.gpkg',
             public_transport = 'tests/data/processed/public_transport.gpkg',
             buildings = 'tests/data/processed/buildings.shp',
             neighborhoods = 'tests/data/processed/neighborhoods.gpkg')
    
if __name__ == '__main__':
    unittest.main()
    
    