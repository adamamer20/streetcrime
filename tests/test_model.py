import unittest

from streetcrime.model import StreetCrime
from streetcrime.space.city import City


class TestModel(unittest.TestCase):
    def test_model_init(self):
        city = City(crs="epsg:32632", city_name="Test City")
        city.load_data(
            roads_file="tests/data/processed/roads.gpkg", buildings_file="tests/data/processed/buildings.shp"
        )
        StreetCrime(space=city)

    def test_model_with_agents(self):
        city = City(crs="epsg:32632", city_name="Test City")
        city.load_data(
            roads_file="tests/data/processed/roads.gpkg", buildings_file="tests/data/processed/buildings.shp"
        )
        model = StreetCrime(space=city)
        model.create_agents(n_agents=100, p_agents={})


if __name__ == "__main__":
    unittest.main()
