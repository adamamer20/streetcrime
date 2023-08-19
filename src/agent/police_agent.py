import mesa_geo as mg
from shapely.geometry import Point
from src.agent.mover import Mover

class PoliceAgent(Mover):
    
    def __init__(self, unique_id, model, geometry, crs):
        super(Mover, self).__init__(unique_id, model, geometry, crs)
        #Position initiated randomly
        random_building_id = self.model.space.get_random_building()
        self.geometry = Point(self.model.space.buildings_df.at[random_building_id, "position"])

    def step(self):
        super().step()
                        