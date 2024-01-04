import unittest
from streetcrime.model import StreetCrime
from streetcrime.space.city import City

class TestModel(unittest.TestCase):
    def test_model_init(self):
        StreetCrime(space = City(crs = 'epsg:7791',
                                 roads = 'tests/data/processed/roads.gpkg',
                                 public_transport = 'tests/data/processed/public_transport.gpkg',
                                 neighborhoods = 'tests/data/processed/neighborhoods.gpkg',
                                 buildings = 'tests/data/processed/buildings.shp'))
    
    def test_model_with_agents(self):
        StreetCrime(space = City(crs = 'epsg:7791',
                                 roads = 'tests/data/processed/roads.gpkg',
                                 public_transport = 'tests/data/processed/public_transport.gpkg',
                                 neighborhoods = 'tests/data/processed/neighborhoods.gpkg',
                                 buildings = 'tests/data/processed/buildings.shp'),
                    num_movers=100)
    
if __name__ == '__main__':
    unittest.main()