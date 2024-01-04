import os.path
import mesa
from streetcrime.model.model import StreetCrime
import pandas as pd
import datetime as dt
import numpy as np

current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]


#Collect agents data
#run_agents = pd.DataFrame()
#step_agents = [agent.data for agent in Milan.space.agents]
#step_agents = pd.DataFrame(step_agents)
#run_agents = pd.concat([run_agents, step_agents])

params = {'p_criminals' : np.arange(0.01, 0.05, 0.01)}

results = mesa.batch_run(StreetCrime,
                        parameters = params,
                        iterations = 1, #After last step
                        max_steps = 10)

results_df = pd.DataFrame(results)

for n_iteration in results_df:
    n_iteration.to_csv(os.path.join(parent_directory, r"outputs\results"+ str(n_iteration) + ".csv"))
    

#Milan.data['crimes'].to_csv(os.path.join(parent_directory, r"outputs\crimes.csv"))
#Milan.data['info_neighborhoods'].to_csv(os.path.join(parent_directory, r"outputs\info_neighborhoods.csv"))
#run_agents.to_csv(os.path.join(parent_directory, r"outputs\run_agents.csv"))

#Milan_df_agents = Milan.datacollector.get_agent_vars_dataframe()
#Milan_df_model = Milan.datacollector.get_model_vars_dataframe()

#Milan_df_agents.to_csv(os.path.join(parent_directory, r"outputs\run_agents.csv"))
#Milan_df_model.to_csv(os.path.join(parent_directory, r"outputs\run_model.csv"))