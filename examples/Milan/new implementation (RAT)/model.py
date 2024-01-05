from time import time

from streetcrime.agents.criminal import Pickpocket, Robber
from streetcrime.agents.police_agent import PoliceAgent
from streetcrime.agents.worker import Worker
from streetcrime.model import StreetCrime
from streetcrime.space.city import City

milan = City(crs="EPSG:32632", city_name="Milan, Italy")

milan.load_data()

model = StreetCrime(space=milan)

model.create_agents(
    p_agents={Worker: 0.85, PoliceAgent: 0.05, Pickpocket: 0.05, Robber: 0},
    n_agents=1000,
)

start_time = time()
model.run_model(days=3, len_step=10)
model.plot()

print("Execution time: " + "--- %s seconds ---" % (time() - start_time))
