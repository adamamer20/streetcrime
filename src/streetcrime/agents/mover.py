import random
from dataclasses import dataclass
from datetime import datetime
from sys import getsizeof
from time import time

import geopandas as gpd
import networkx as nx
import numpy as np  # Generate working hours
import osmnx as ox
import pandas as pd
import scipy
from mesa_frames.agent import GeoAgentDF


@dataclass
class MoverParams:
    """
    It contains the parameters for the Mover class

    Attributes:
    ----------
    walking_speed : float
        The speed of the Mover when walking (m/s). Default: 1.9
    car_use_threshold : float
        *FUTURE IMPLEMENTATION*. The maximum distance the Mover can walk (m) before driving. Default: 5000
    act_decision_rule : str
        *FUTURE IMPLEMENTATION*. The rule used to decide which activity to go to. Default: "buildings, weights = 1/df.geometry.distance(agent.geometry)"

    """

    walking_speed: float = 1.9  # m/s
    # car_use_threshold : float = 5000 #m
    # car : bool = True
    # act_decision_rule : str = "buildings, weights = 1/df.geometry.distance(agent.geometry)"

class Mover(GeoAgentDF):
    """The Mover class, is the base class for all agents in the simulation.
    It is able to move around the city using the road network.
    It has a unique id and a geometry (Point) that represents his position in the city. He also has a status that describes what he is currently doing.
    At every step, the mover moves but only if he is travelling. If he's free, he gets a random activity to go to.

    Attributes:
    ----------
    dtypes : dict[str, str]
        The attributes of the Agent as a dictionary of columns and data types. It contains:
        - node : int32
            -- The id of the node where the Mover is.
        - status : str
            -- The status of the Mover. Can be "free", "transport", "busy", "home"
        - destination : int32
            -- The id of the building where the Mover is going to.
        - activity_end_time : datetime64[s]
            -- The time when the Mover ends his activity.
        - last_neighborhood : int64
            -- *FUTURE IMPLEMENTATION*. The id of the neighborhood where the Mover was in the previous step.
        - network : int64
            -- *FUTURE IMPLEMENTATION*. The id of the network where the Mover is.
        - car : bool
            -- *FUTURE IMPLEMENTATION*. Whether the Mover is using a car or not.
    params : MoverParams
        It contains the parameters for the mover class
    """

    params: MoverParams = MoverParams()
    dtypes: dict[str, str] = {
        "node": "int32",
        "status": "str",
        "destination": "int32",
        "activity_end_time": "datetime64[s]",
        #'last_neighborhood' : 'int64',
        #'network' : 'int64',
        #'car' : bool,
    }

    @classmethod
    def __init__(cls, params: MoverParams = MoverParams()) -> None:
        """
        Parameters:
        ----------
        walking_speed : float
            The speed of the Mover when walking (m/s). Default: 1.9
        car_use_threshold : float
            *FUTURE IMPLEMENTATION*. The maximum distance the Mover can walk (m) before driving. Default: 5000
        act_decision_rule : str
            *FUTURE IMPLEMENTATION*. The rule used to decide which activity to go to. Default: "buildings, weights = 1/df.geometry.distance(agent.geometry)"
        """
        super().__init__()
        # Initialize parameters
        cls.params = params
        # cls.car_use_threshold = car_use_threshold
        # cls.act_decision_rule = act_decision_rule

        # Initialize attributes
        cls.model.agents.loc[cls.mask, "status"] = "free"
        cls.model.agents.loc[cls.mask, "node"] = cls.model.space.get_random_nodes(
            n=cls.mask.sum()
        )
        # cls.model.agents.loc[cls.mask, "car"] = False

    @classmethod
    def gen_attribute(
        cls,
        limits: list[float | datetime] = None,
        attribute_type: str = "float",
        distribution: str = "normal",
        mean: float = None,
        sd: float = None,
        a: float = None,
        loc: float = None,
        scale: float = None,
        n: int = 1,
    ) -> int | float | pd.Series:
        """
        Returns a Pandas Series of random values for the attribute based on specified parameters.

        Parameters:
        ----------
        attribute_type : str, default="float"
            the type of the attribute to generate. Can be "float", "datetime_fixed", "datetime_variable"
        distribution : str, default="normal"
            the distribution used to generate. Can be "normal", "uniform", "skewnorm"
        n : int, default=1
            the number of values to generate

        Returns:
        ----------
        pd.Series
        """
        if distribution == "normal":
            attribute_values = np.random.normal(mean, sd, n)
            if limits is not None:
                attribute_values = np.clip(attribute_values, limits[0], limits[1])

        elif distribution == "uniform":
            attribute_values = np.random.uniform(limits[0], limits[1], n)

        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

        return attribute_values

        '''#TODO: implement all features in the refactored function
            def gen_attribute(cls,
                        agents_mask : pd.Series,
                        limits : list[float | datetime] = None,
                        attribute_type : str = "float", 
                        distribution = "normal",
                        mean : float = None,
                        sd : float = None,
                        a: float = None,
                        loc: float = None,
                        scale: float = None,
                        next_day : float = False) -> float | datetime | list[int]:
            """Returns a random value for the attribute `attribute_name` based on the specified type, and mean/sd/min/max specified in params.

            Parameters:
            ----------
            attribute_name : str
                the name of the attribute to generate 
            attribute_type : str, default="float"
                the type of the attribute to generate. Can be "float", "datetime_fixed", "datetime_variable"
            distribution : str, default="normal"
                the distribution used to generate. Can be "normal", "uniform"
                
            Returns:
            ----------
            attribute_value : float | datetime | list[int]
            """
            match distribution:
                case "normal":
                    attribute_value = np.random.normal(mean, sd)
                    if limits is not None:
                        while attribute_value < limits[0] or attribute_value > limits[1]:
                            attribute_value = np.random.normal(mean, sd)
                    #Return based on the time
                    if attribute_type in ["datetime_variable", "datetime_fixed", "timedelta"]:
                        attribute_value = _proper_time_type(time = attribute_value, 
                                                            attribute_type = attribute_type, 
                                                            len_step = cls.model.len_step,
                                                            datetime = cls.model.datetime,
                                                            next_day = next_day) 
                case "uniform":
                    attribute_value = np.random.uniform(limits[0], limits[1])
                case "skewnorm":
                    attribute_value = scipy.stats.skewnorm.rvs(a, loc, scale)
            return attribute_value'''

    @classmethod
    def step(cls):
        """The step method of the Mover."""
        # Check which cls.model.agents have finished their activity and update their status
        mask = (
            cls.mask
            & (cls.model.agents["status"] == "busy")
            & (cls.model.agents["activity_end_time"] <= cls.model.datetime)
        )
        cls.model.agents.loc[mask, "status"] = "free"

        # Set activity for cls.model.agents who are free and do not have a destination
        cls._set_activity(
            cls.mask
            & (cls.model.agents["status"] == "free")
            & (cls.model.agents["destination"].isna())
        )

        # Find paths for cls.model.agents without path and with a destination
        cls._set_path(
            cls.mask
            & (cls.model.agents["destination"].notna())
            & (  # those who have a destination
                (
                    cls.model.agents.index.isin(
                        cls.model.paths[cls.model.paths["ancestor"].isna()].index
                    )
                )
                & (  # and do not have a path
                    cls.model.agents.destination != cls.model.agents.node
                )
            )
        )  # and are not already at the node

        # Move agents #TODO: implement other modes of transport
        moving_mask = cls.model.paths.index.isin(cls.model.agents[cls.mask].index)
        cls.model.paths.loc[moving_mask, "travelled_distance"] = cls.model.paths[
            moving_mask
        ]["travelled_distance"] + (cls.model.len_step * 60 * cls.params.walking_speed)

        # Mark cls.model.agents who have reached their destination
        arrived_mask_paths = cls.model.paths.index.isin(
            cls.model.agents[cls.mask].index
        ) & (  # those who have a path
            cls.model.paths["travelled_distance"]
            >= cls.model.paths["length"].groupby(level=0).transform("last")
        )  # and have reached the end of the path
        arrived_mask_agents = cls.model.agents.index.isin(
            cls.model.paths[arrived_mask_paths].index
        ) | (  # those who have a path and have reached the end reindexed for model.agents
            cls.model.agents["destination"] == cls.model.agents["node"]
        )  # or those who are already at the node
        cls._set_arrival(arrived_mask_agents, arrived_mask_paths)

        # Update the geometry of those who have not reached their destination
        moving_mask = cls.model.agents.index.isin(
            cls.model.paths[cls.model.paths.ancestor.notna()].index
        )
        cls._set_geometry(moving_mask)

        '''#For Workers and PoliceAgents, add their visit to info_neighborhoods
        if column is not None:
            neighborhood_id = cls.cls.model.space.find_neighborhood_by_pos(cls.geometry)
            if (cls.last_neighborhood != neighborhood_id) & (neighborhood_id is not None):
                cls.cls.model.info_neighborhoods.loc[:, column] = cls.cls.model.info_neighborhoods[column].astype(float)
                cls.cls.model.info_neighborhoods.loc[(0, neighborhood_id), column] += 1
                cls.cls.model.info_neighborhoods.loc[:, column] = cls.cls.model.info_neighborhoods[column].astype(pd.SparseDtype(float, np.nan))
                cls.last_neighborhood = neighborhood_id"""'''

    @classmethod
    def _set_activity(cls, agents_mask):
        """
        The method sets an activity for the agents in the mask.
        """
        # If agents are free and do not have a destination, they get an activity
        if (
            cls.model.datetime.hour >= cls.model.day_act_start
            and cls.model.datetime.hour <= cls.model.day_act_end
        ):
            act_time = "open_day"
        else:
            act_time = "open_night"
        cls.model.agents.loc[
            agents_mask, "destination"
        ] = cls.model.space.get_random_nodes("activity", act_time, n=agents_mask.sum())

    @classmethod
    def _set_path(cls, agents_mask):
        """
        The method sets the path for agents in the mask
        """
        # Initialize columns
        cls.model.agents.loc[agents_mask, "status"] = "transport"
        cls.model.agents.loc[agents_mask, "travelled_distance"] = 0

        # If possible, retrieve from paths, exploding to divide into edges
        paths = (
            pd.merge(
                cls.model.agents.loc[
                    agents_mask, ["node", "destination"]
                ].reset_index(),
                cls.model.space.paths,
                on=["node", "destination"],
                how="left",
            )
            .set_index("id")
            .explode("path")
            .path
        )
        paths = paths.astype("int")
        paths.name = "new_ancestor"
        cls.model.paths = pd.merge(
            cls.model.paths, paths, left_index=True, right_index=True, how="left"
        )

        agents_mask = (
            cls.model.paths.ancestor.isna() & cls.model.paths.new_ancestor.notna()
        )
        cls.model.paths.loc[agents_mask, "ancestor"] = cls.model.paths.loc[
            agents_mask, "new_ancestor"
        ]
        cls.model.paths.drop("new_ancestor", axis=1, inplace=True)
        cls.model.paths.loc[agents_mask, "successor"] = (
            cls.model.paths[agents_mask].groupby(level=0)["ancestor"].shift(-1)
        )

        # Find the cumulative length of edges
        cls.model.paths.loc[agents_mask, "length"] = (
            cls.model.paths.loc[agents_mask]
            .drop(columns="length")
            .merge(
                cls.model.space.roads_edges["length"],
                left_on=["ancestor", "successor"],
                right_index=True,
                how="left",
            )["length"]
        )
        cls.model.paths.loc[agents_mask, "length"] = (
            cls.model.paths.loc[agents_mask, "length"].groupby(level=0).cumsum()
        )
        cls.model.paths.loc[agents_mask, "travelled_distance"] = 0
        # Drop the last ancestor occurence because it is the destination
        cls.model.paths = cls.model.paths[
            ~(cls.model.paths.ancestor.notna() & cls.model.paths.successor.isna())
        ]
        # cls.model.agents.loc[:, ['ancestor', 'successor']] = cls.model.agents[['ancestor', 'successor']].astype(int)

    @classmethod
    def _set_arrival(cls, agents_mask, paths_mask):
        """The method set arrival for agents in the mask for model.agents and model.paths"""
        cls.model.paths.loc[paths_mask] = np.nan
        cls.model.paths = cls.model.paths[
            (paths_mask & (~cls.model.paths.index.duplicated())) | (~paths_mask)
        ]

        # Update columns
        at_home_mask = agents_mask & (
            cls.model.agents["destination"] == cls.model.agents["home"]
        )
        cls.model.agents.loc[at_home_mask, "status"] = "home"
        at_work_mask = agents_mask & (
            cls.model.agents["destination"] == cls.model.agents["work"]
        )
        cls.model.agents.loc[at_work_mask, "status"] = "home"
        act_mask = agents_mask & (~at_home_mask) & (~at_work_mask)
        cls.model.agents.loc[act_mask, "status"] = "busy"
        cls.model.agents.loc[act_mask, "activity_end_time"] = pd.to_datetime(
            cls.model.datetime
            + pd.to_timedelta(
                cls.model.activity_len_distr.rvs(size=act_mask.sum()), unit="m"
            )
        )
        cls.model.agents.loc[agents_mask, "node"] = cls.model.agents.loc[
            agents_mask, "destination"
        ]
        cls.model.agents.loc[
            agents_mask,
            [
                "destination",
                "length",
                "ancestor",
                "successor",
                "travelled_distance",
                "geometry",
            ],
        ] = np.nan

        # Keep only one occurrence for arrived agents
        cls.model.agents = cls.model.agents[
            (agents_mask & (~cls.model.agents.index.duplicated())) | (~agents_mask)
        ]

    @classmethod
    def _set_geometry(cls, agents_mask):
        """The method sets the geometry for agents in the mask"""
        current_edge = (
            cls.model.paths[
                cls.model.paths["travelled_distance"] <= cls.model.paths["length"]
            ]
            .groupby(level=0)
            .first()
        )
        cls.model.agents.loc[agents_mask, "node"] = current_edge["ancestor"].reindex(
            cls.model.agents.loc[agents_mask].index
        )
        current_edge = current_edge.merge(
            cls.model.space.roads_edges["geometry"],
            left_on=["ancestor", "successor"],
            right_index=True,
            how="left",
        )
        current_edge = current_edge.set_geometry("geometry")
        cls.model.agents.loc[agents_mask, "geometry"] = current_edge.interpolate(
            current_edge["travelled_distance"] / current_edge["length"], normalized=True
        )
