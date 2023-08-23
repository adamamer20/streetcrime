import mesa_geo as mg
from shapely.geometry import Point
from src.agent.mover import Mover

class PoliceAgent(Mover):
    attributes ={
        'policeman' : True
    }
    
    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
        self.data['status'] = "free"
        random_building_id = self.model.space.get_random_building()
        self.geometry = Point(self.model.space.buildings.at[random_building_id, "geometry"].centroid.coords[0])
        
    def step(self):
        super().step()
                        