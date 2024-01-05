# streetcrime: a python ABM package for urban crime

Urban crime, with its complex local interactions and spatial dynamics, is an excellent candidate for Agent-Based Modeling (ABM). The `streetcrime` package is designed as a robust, user-friendly ABM tool for urban crime analysis. It facilitates the testing of various crime theories, compares them with real crime data, and evaluates crime reduction strategies. Future versions will include Machine Learning (ML) features for parameter tuning. The package integrates [`osmnx`](https://github.com/gboeing/osmnx) for urban data and [`mesa`](https://github.com/projectmesa/mesa) (along with[ `mesa-frames`](https://github.com/adamamer20/mesa-frames)) for the modeling framework.

## Installation

### Prerequisites
Before installing `streetcrime`, ensure that mesa-frames is installed. As it's not yet available on PyPi or conda-forge, follow the installation instructions [here](https://github.com/adamamer20/mesa-frames#installation).

### Installation Steps
1. **Clone the GitHub Repository**
    ```bash
    git clone https://github.com/adamamer20/streetcrime.git
    cd streetcrime
    ```

2. **Installation**
   
   a. *Installing in a Conda Environment*
      ```bash
      conda activate myenv
      pip install -e .
      ```

   b. *Installing in a Python Virtual Environment*
      ```bash
      source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
      pip install -e .
      ```

### Usage
*Note: `streetcrime` is in early development stages; expect changes and potential breaking updates. Feedback and issue reports are welcomed.*

- You can find the API Documentation [here](https://adamamer20.github.io/streetcrime/api)

- You can find a simple usage script [here](https://github.com/adamamer20/streetcrime/blob/main/examples/Milan/Simple%20RAT/model.py)

### Key Components
#### City
The City class represents the geographical space for agent interactions, created using a CRS and an OpenStreetMap query. It features an advanced road network and pre-calculated shortest paths for efficiency. Buildings are categorized into homes, workplaces, and potential activities. They are also categorized as being open or closed during night-time. You can find the categorization [here](https://github.com/adamamer20/streetcrime/blob/main/src/streetcrime/space/city.py#L360C13-L360C33).

*Note: Data for large cities may take longer to download; it's recommended to start with smaller areas for initial tests.

```python
from streetcrime.space.city import City
from streetcrime.model import StreetCrime
from streetcrime.agents.criminal import Pickpocket, Robber
from streetcrime.agents.police_agent import PoliceAgent
from streetcrime.agents.worker import Worker

milan = City(crs="EPSG:32632", city_name="Milan, Italy")
milan.load_data()
```

#### Agents
Here is a diagram which shows the currently supported agents:

<img src="https://github.com/adamamer20/streetcrime/blob/main/docs/images/classes.png" width="30%"/>

New agents should be created from existing classes. For example, a House Thief could inherit from the Criminal class.


##### Base Classes
- **Mover**: This is the foundational class for all moving agents within the model. The Mover class is equipped with the ability to navigate the city using its road network. Key attributes include a unique identifier, a geometric point representing the agent's location, and a status indicating their current activity. The class also handles the selection of random activities for agents and manages their movement to these locations. Upon reaching a destination, a countdown begins, determining when the agent can move again. Future enhancements will include diverse rules for activity selection.

- **InformedMover**: This class represents an advanced version of Mover, planned for future implementation. InformedMover agents will have access to a subset of global model information, such as recent crimes or heavily trafficked areas, based on a defined threshold. This feature allows for more sophisticated decision-making and movement patterns.

- **Resident**: Building on the InformedMover, the Resident class has a designated home location and a specified rest period, during which the agent must stay at home. This class simulates the daily life patterns of city inhabitants.

##### Deployable Classes

- **PoliceAgent**: A subclass of InformedMover, the PoliceAgent functions as a guardian within the city. Their primary role is crime prevention; when in close proximity to a Criminal agent, they can inhibit criminal activities, thus playing a crucial role in the model's representation of law enforcement and public safety dynamics.

- **Worker**: Deriving from the Resident class, Workers are typical inhabitants of the city with defined routines, including work locations and schedules. They represent potential targets for criminal agents, possessing a variable attribute of 'crime_attractiveness' which influences their likelihood of being victimized. This class illustrates the daily patterns of city residents and their interactions with other agent types.

- **Criminal (Pickpocket and Robber)**: These agents, branching from the Resident class, are central to the simulation of criminal activities within the city. Criminals actively seek opportunities to commit crimes, selecting their targets based on perceived opportunities and victim attributes. The Pickpocket and Robber subclasses differ in their operational tactics and the criteria for successful criminal acts. For instance, Pickpockets thrive in crowded settings, while Robbers may face additional challenges, such as potential resistance from Workers.

### Model
After initliazing a city of choice and having loaded data, running involves initializing the agents and executing the simulation. 

```python
model = StreetCrime(milan)
model.create_agents(
    p_agents={Worker: 0.85, PoliceAgent: 0.05, Pickpocket: 0.05, Robber: 0.05}, 
    n_agents=1000
)

#Run the model for 3 days considering time steps of 10 minutes
model.run_model(days=3, len_step=10)
```

Users can also save the simulation state for analysis and resume from saved states. Graphs of crimes and agents can be generated.

```python
#Save the state of the simulation
model.save_state()

#Plot the model at current state
model.plot()
```

## What's Next?
- Enabling information access for InformedMover.
- Incorporating public transport routes data.
- Integrating diverse criminological theories (eg. ).
- Enhancing visualization capabilities, including live or post-simulation video generation.
- Strengthening integration with mesa and mesa-frames.
- Introducing various criminal types (eg. House Thiefs).
- Implementing ML for parameter optimization.

