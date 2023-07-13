import mesa

from src.space.building import Building
from src.agent.resident import Resident
from src.space.road_network import RoadNetwork

class ClockElement(mesa.visualization.TextElement):
    def __init__(self):
        super().__init__()
        pass

    def render(self, model):
        return f"Day {model.day}, {model.hour:02d}:{model.minute:02d}"


def agent_draw(agent):
    portrayal = dict()
    portrayal["color"] = "White"
    if isinstance(agent, RoadNetwork):
        portrayal["color"] = "#D08004"
    elif isinstance(agent, Building):
        portrayal["color"] = "Grey"
        '''
        if agent.function is None:
             portrayal["color"] = "Grey"
         elif agent.function == 1.0:
             portrayal["color"] = "Blue"
         elif agent.function == 2.0:
             portrayal["color"] = "Green"
         else:
             portrayal["color"] = "Grey"
        '''
    elif isinstance(agent, Resident):
        if agent.status == "home":
            portrayal["color"] = "Green"
        elif agent.status == "work":
            portrayal["color"] = "Blue"
        elif agent.status == "transport":
            portrayal["color"] = "Red"
        else:
            portrayal["color"] = "Grey"
        portrayal["radius"] = "5"
        portrayal["fillOpacity"] = 1
    return portrayal


clock_element = ClockElement()
status_chart = mesa.visualization.ChartModule(
    [
        {"Label": "status_home", "Color": "Green"},
        {"Label": "status_work", "Color": "Blue"},
        {"Label": "status_traveling", "Color": "Red"},
    ],
    data_collector_name="datacollector",
)
