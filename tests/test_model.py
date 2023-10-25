import unittest
from src.model import StreetCrime
from src.space.city import City

class TestModel(unittest.TestCase):
    def test_model_init(self):
        StreetCrime(space = City(crs = 'epsg:7791',
                                 roads = 'tests/data/processed/roads.gpkg',
                                 public_transport = 'tests/data/processed/public_transport.gpkg',
                                 neighborhoods = 'tests/data/processed/neighborhoods.gpkg',
                                 buildings = 'tests/data/processed/buildings.shp'))
    
if __name__ == '__main__':
    unittest.main()