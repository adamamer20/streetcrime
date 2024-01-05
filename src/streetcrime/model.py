import math
import os.path
from datetime import date, datetime, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesa_frames.model import ModelDF
from scipy import stats

from streetcrime.agents.mover import Mover
from streetcrime.space.city import City


class StreetCrime(ModelDF):
    """The StreetCrime class contains the model of the simulation.

    Parameters
    ----------
    space : City
        The city where the simulation takes place.
    activity_len_distr : scipy.stats.lognorm
        The lognorm distribution of the length of an activity.
    datetime : datetime
        The current datetime of the simulation.
    len_step : int | None
        The length of each step in minutes.
    day_act_start : int
        The hour when activities open during the day become available .
    day_act_end : int
        The hour when activities open during the day become unavailable .
    mean_activity_length : float
        The mean length of an activity in minutes.
    sd_activity_length : float
        The standard deviation of the length of an activity in minutes.
    paths : pd.DataFrame
        The dataframe containing current paths for agents in movement.
    crimes : gpd.GeoDataFrame
        The dataframe containing crimes committed during the simulation.
    """

    def __init__(
        self,
        space: City,
        data_collection: str = "2d",  # TODO: implement different data collection frequencies (xw, xd, xh, weekly, daily, hourly)
        transport: str
        | list[
            str
        ] = "on_foot",  # TODO: implement other modes of transport 'car', 'public_transport'
        crime_theory: str = "RAT",  # TODO: implement other crime theories
        len_step: int = 10,  # minutes
        start_datetime: datetime = datetime(
            date.today().year, date.today().month, date.today().day, 5, 30
        ),
        day_act_start: int = 8,
        day_act_end: int = 19,
        mean_activity_length: float = 60,  # minutes
        sd_activity_length: float = 20,  # minutes
    ):
        """Initializes the StreetCrime class.

        Parameters
        -----------
        space : City
            The city where the simulation takes place.
        data_collection : str
            *FUTURE IMPLEMENTATION*. The frequency of the data collection. It can be 'xd', 'xd', 'xh', 'weekly', 'daily', 'hourly'. Default: '2d'
        transport : str | list[str]
            *FUTURE IMPLEMENTATION*. The mode of transport of the agents. It can be 'on_foot', 'car', 'public_transport'. Default: 'on_foot'
        crime_theory : str
            *FUTURE IMPLEMENTATION*. The crime theory used in the simulation. It can be 'RAT', 'CPT', 'SCT'. Default: 'RAT'
        start_datetime : datetime
            The datetime when the simulation starts. Default: datetime(date.today().year, date.today().month, date.today().day, 5, 30)
        day_act_start : int
            The hour when activities open during the day become available . Default: 8
        day_act_end : int
            The hour when activities open during the day become unavailable . Default: 19
        mean_activity_length : float
            The mean length of an activity in minutes. Activity length is distributed as a lognorm. Default: 60
        sd_activity_length : float
            The standard deviation of the length of an activity in minutes. Activity length is distributed as a lognorm. Default: 20
        """
        super().__init__(space=space)

        # Initialize model attributes
        self.activity_len_distr = stats.lognorm(
            s=np.sqrt(np.log(sd_activity_length**2 / mean_activity_length**2 + 1)),
            scale=math.exp(
                np.log(mean_activity_length)
                - 0.5 * np.log(sd_activity_length**2 / mean_activity_length**2 + 1)
            ),
        )
        self.start_datetime = start_datetime
        self.datetime = start_datetime
        self.day_act_start = day_act_start
        self.day_act_end = day_act_end
        self.len_step: int | None = None
        self.paths: pd.DataFrame | None = None

        self.crimes = gpd.GeoDataFrame(
            columns=[
                "type",
                "datetime",
                "geometry",
                "neighborhood",
                "victim",
                "criminal",
                "witnesses",
                "successful",
            ]
        )

    def create_agents(
        self,
        n_agents: int,
        p_agents: dict[type[Mover], float],
    ) -> None:
        """Creates agents of the model.

        Parameters
        ----------
        n_agents : int
            The number of agents to create.
        p_agents : dict[type[Mover], float]
            A dictionary with the type agent and the proportion out of a 100.
        """
        # Create movers
        super().create_agents(n_agents, p_agents)
        self.paths = pd.DataFrame(
            np.nan,
            index=self.agents.index,
            columns=["ancestor", "successor", "length", "travelled_distance"],
        )
        # self.agents_info = self._create_agents_info()

    def run_model(self, days: int, len_step: int) -> None:
        """
        Runs the model for a number of days with a given length of step in minutes.

        Parameters
        ----------
        days : int
            The number of days to run the model.
        len_step : int
            The length of each step in minutes."""
        self.len_step = len_step
        n_steps = int(days * 24 * 60 / len_step)

        for _ in range(n_steps):
            # Advance time
            self.datetime = self.datetime + timedelta(minutes=len_step)
            super().step(merged_mro=True)

    def save_state(self):
        """Saves the state of the model (agents, crimes, path and model info) in the outputs/runs folder.
        Allows to load the state of the model later.
        """
        if not os.path.exists(f"outputs/runs/{self.unique_id}"):
            os.makedirs(f"outputs/runs/{self.unique_id}")
        self.agents.to_file(f"outputs/runs/{self.unique_id}/agents.gpkg")
        with open(f"outputs/runs/{self.unique_id}/model.txt", "w") as f:
            f.write(
                f"""
        Start Datetime: {self.start_datetime}
        Length of step: {self.len_step}
        Space: {self.space.city_name}
        Number of agents: {len(self.agents)}
        Current datetime: {self.datetime}
        """
            )
        self.crimes.datetime = self.crimes.datetime.astype(str)
        self.crimes.to_file(f"outputs/runs/{self.unique_id}/crimes.gpkg")
        self.paths.to_csv(f"outputs/runs/{self.unique_id}/paths.csv")

    def plot(
        self,
        roads=True,
        buildings=True,
        agents=True,
        boundaries=True,
        crimes=True,
        save=True,
    ):
        """Plots the current state of the model.

        Parameters
        ----------
        roads : bool, optional
            If True plots road network, by default True
        buildings : bool, optional
            If True plots buildings as grey, by default True
        agents : bool, optional
            If True plots agents as green points, by default True
        boundaries : bool, optional
            if True plots boundaries of the city, by default True
        crimes : bool, optional
            If True plots committed crimes as red points, by default True
        save : bool, optional
            If True saves the plots in output/runs/plots, by default True
        """
        self.save_state()

        alpha = 1 if boundaries else 0
        ax = gpd.read_file(
            f"outputs/city/{self.space.city_name.split(',')[0]}_boundaries.gpkg"
        ).plot(alpha=alpha)

        if roads:
            self.space.roads_nodes.plot(ax=ax, color="black", markersize=1)
            self.space.roads_edges.plot(ax=ax, color="black")

        if buildings:
            gpd.read_file(
                f"outputs/city/{self.space.city_name.split(',')[0]}_buildings.gpkg"
            ).plot(ax=ax, color="grey")

        if agents:
            (
                gpd.GeoDataFrame(
                    self.agents[self.agents.geometry.isna()]
                    .drop("geometry", axis="columns")
                    .merge(
                        self.space.roads_nodes.geometry,
                        left_on="node",
                        right_index=True,
                    )
                ).plot(ax=ax, color="green")
            )
            self.agents[self.agents.geometry.notna()].geometry.plot(
                ax=ax, color="green"
            )

        if crimes:
            self.crimes.plot(ax=ax, color="red")

        ax.set_axis_off()
        if save:
            plt.savefig(f"outputs/runs/{self.unique_id}/plot.png")
        else:
            plt.show()

    def load_state(self, unique_id: str):
        self.unique_id = unique_id
        self.agents = gpd.read_file(f"outputs/runs/{self.unique_id}/agents.gpkg")
        self.agents.set_index("id", inplace=True)
        self.crimes = gpd.read_file(f"outputs/runs/{self.unique_id}/crimes.gpkg")
        self.crimes.datetime = pd.to_datetime(self.crimes.datetime)
        self.paths = pd.read_csv(f"outputs/runs/{self.unique_id}/paths.csv")

    ''' #TODO: implement data collector in mesa_frames
    
    def get_data(self) -> dict[str, pd.DataFrame]:
        """
        Returns a dictionary with dataframes of the data of the model, the parameters  and the agents.

        
        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary with dataframes of the data of the model, the parameters  and the agents.
        """
        data = pd.DataFrame([{
            'seed' : self.seed,
            'run_id': self.run_id,
            'model_params' : self.model_params,
            'agents_params' : self.agents_params,
            'movers': self.agents,
            'model_params': None,
            'agents_params': None
        }]).set_index('run_id')
        if isinstance(self.model_params, dict):
            data.at[self.run_id, 'model_params'] = pd.DataFrame([self.model_params])
        else:
            data.at[self.run_id, 'model_params'] = pd.DataFrame([self.model_params.value])
        if isinstance(self.agents_params, dict):
            data.at[self.run_id, 'agents_params'] = pd.DataFrame([self.agents_params])
        else:
            data.at[self.run_id, 'agents_params'] = pd.DataFrame([self.agents_params.value])
        return data
        
    def _initialize_data_collection(self, how="2d") -> None:
        """Initializes the data collection of the model.

        Parameters
        ----------
        how : str
            The frequency of the data collection. It can be 'xd', 'xd', 'xh', 'weekly', 'daily', 'hourly'.
        """
        # TODO: finish implementation of different data collections
        if how == "2d":
            categories = [
                "yesterday_visits",
                "today_visits",
                "run_visits",
                "yesterday_crimes",
                "today_crimes",
                "run_crimes",
                "yesterday_police",
                "today_police",
                "run_police",
            ]
        return {key: 0 for key in categories}'''

    '''#TODO: implement information in refactored version
    def _update_information(self) -> None:
        """Updates 'yesterday_crimes' and 'yesterday_visits' columns of the self.buildings with data
        collected in today + '_crimes' and today + '_visits' columns of the self.neighborhoods dataframe.
        """
        self.movers_info.loc[:, 'run_visits'] += self.movers_info.loc[:, 'today_visits'] - 1
        self.movers_info.loc[:, 'run_crimes'] += self.movers_info.loc[:, 'today_crimes'] - 1
        self.movers_info.loc[:, 'run_police'] += self.movers_info.loc[:, 'today_police'] - 1
        self.movers_info.loc[:, 'yesterday_visits'] = self.movers_info.loc[:, 'today_visits'] 
        self.movers_info.loc[:, 'yesterday_crimes'] = self.movers_info.loc[:, 'today_crimes'] 
        self.movers_info.loc[:, 'yesterday_police'] = self.movers_info.loc[:, 'today_police'] 
        self.movers_info.loc[:, 'yesterday_visits'] -= 1
        self.movers_info.loc[:, 'yesterday_crimes'] -= 1
        self.movers_info.loc[:, 'yesterday_police'] -= 1
        self.movers_info.update(self.info_neighborhoos)
        self.space.buildings.drop(['yesterday_visits', 
                             'yesterday_crimes',
                             'yesterday_police',
                             'run_visits', 
                             'run_crimes',
                             'run_police'], inplace=True, axis='columns')
        copy =self.movers_info.copy()
        copy = copy[['yesterday_visits', 
                    'yesterday_crimes',
                    'yesterday_police',
                    'run_visits', 
                    'run_crimes',
                    'run_police']]
        self.space.buildings = self.space.buildings.merge(
            copy, left_on='neighborhood', right_index = True)
        
        #Add today columns
        self.movers_info = self.__set_date_columns(self.datetime.date(), self.movers_info)
        
        self.movers_info = self.movers_info.astype(pd.SparseDtype(float, np.nan))
        def _create_agents_info(self) -> pd.DataFrame():
        idx = pd.MultiIndex.from_product([self.agents.id, self.space.neighborhoods.index], names=['mover_id', 'neighborhood_id'])
        columns = ['yesterday_visits', 'today_visits', 'run_visits', 
                   'yesterday_crimes', 'today_crimes', 'run_crimes',
                   'yesterday_police', 'today_police', 'run_police']
        agents_info = pd.DataFrame(1, index = idx, columns = columns)
        
        #TO KEEP EVERY DAY (Adding today columns)
        #agents_info = self.__set_date_columns(self.datetime.date(), agents_info)
        
        return agents_info'''
