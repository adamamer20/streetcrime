import mesa_geo as mg
import mesa
import pyproj
from shapely.geometry import Point
from src.agents.informed_mover import InformedMover

class PoliceAgent(InformedMover):
    guardian = True
    act_decision_rule = "yesterday_crimes * run_crimes"
    
    def __init__(self,
                 unique_id : int, 
                 model : mesa.Model, 
                 geometry : Point, 
                 crs : pyproj.CRS,
                 walking_speed : float = None,
                 driving_speed : float= None,
                 car_use_threshold : float = None,
                 sd_activity_end : float = None,
                 mean_activity_end : float = None,
                 car : bool = None,
                 act_decision_rule : str = None,
                 p_information : float = None,
                 guardian : bool = None):
        super().__init__(unique_id, 
                         model, 
                         geometry, 
                         crs, 
                         walking_speed, 
                         driving_speed, 
                         car_use_threshold, 
                         sd_activity_end, 
                         mean_activity_end,
                         car,
                         act_decision_rule,
                         p_information)
        
        if guardian is not None:
            self.guardian = guardian
        self.status = 'free'
        random_building_id = self.model.space.get_random_building()
        self.geometry = self.model.space.buildings.at[random_building_id, "geometry"].centroid
        
    def step(self):
        super().step()
                        