import os.path
import mesa
from src.model.model import StreetCrime
import pandas as pd
current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
current_directory = parent_directory

Milan = StreetCrime()

#mesa.batch_run(Milan,
#               parameters={
 
#                  params['movers']['Criminal']: [0.1, 0.2, 0.3, 0.4, 0.5],
#               })
# Running the model

run_agents = pd.DataFrame()
for _ in range(Milan.params['n_steps']):
    step_agents = [agent.data for agent in Milan.space.agents]
    step_agents = pd.DataFrame(step_agents)
    run_agents = pd.concat([run_agents, step_agents])
    Milan.data['step_counter'] += 1
    Milan.step()
    Milan.datacollector.collect(Milan)

Milan.data['crimes'].to_csv(os.path.join(parent_directory, r"outputs\crimes.csv"))
Milan.data['info_neighborhoods'].to_csv(os.path.join(parent_directory, r"outputs\info_neighborhoods.csv"))
run_agents.to_csv(os.path.join(parent_directory, r"outputs\run_agents.csv"))

#Milan_df_agents = Milan.datacollector.get_agent_vars_dataframe()
Milan_df_model = Milan.datacollector.get_model_vars_dataframe()

#Milan_df_agents.to_csv(os.path.join(parent_directory, r"outputs\run_agents.csv"))
Milan_df_model.to_csv(os.path.join(parent_directory, r"outputs\run_model.csv"))