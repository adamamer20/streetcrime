import mesa_geo as mg
from shapely.geometry import Point
from src.agents.informed_mover import InformedMover

class PoliceAgent(InformedMover):
    attributes ={
        'policeman' : True
    }
    params = {
        "act_decision_rule": "yesterday_crimes * run_crimes"
    }
    
    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
        self.data['status'] = "free"
        random_building_id = self.model.space.get_random_building()
        self.geometry = Point(self.model.space.buildings.at[random_building_id, "geometry"].centroid.coords[0])
        
    def step(self):
        super().step()
                        