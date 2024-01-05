from dataclasses import dataclass

import numpy as np
import pandas as pd

from streetcrime.agents.resident import Resident, ResidentParams


@dataclass
class CriminalParams(ResidentParams):
    """The CriminalParams class is a dataclass that contains the parameters of the Criminal class.

    Attributes
    ----------
    opportunity_awareness : float
        The distance in meters for which the Criminal can find victims to commit a crime. Default: 300
    crowd_deterrance : float
        The probability of the Criminal to be successfull in his crime is reduced by this factor for each additional Worker in the vicinity. Default: 0.01
    crime_motivation_rule : str
        The rule used to assign the crime motivation attribute to the Criminal. Default: "uniform"
    """

    opportunity_awareness: int = 150
    crowd_effect: float = 0.01
    crime_motivation_rule: str = "uniform"
    # home_decision_rule: str = "neighborhoods, weights = df.prop * (1/df.mean_income)"


class Criminal(Resident):
    """The Criminal class is a subclass of Resident. Criminals travel around the city and commit crimes if they find opportunities.

    Attributes
    ----------
    dtypes : dict[str, str]
        The attributes of the Agent as a dictionary of columns and data types. It contains:
        - crime_motivation : float16
            -- The probability of the Criminal to be successfull in his crime
        - successfull_crime : bool
            -- Whether the Criminal was successfull in his crime
        - victim : int64
            -- The id of the victim of the crime
        - close_agents : object
            -- The number of Workers in the vicinity of the Criminal

    params : CriminalParams
        The parameters of the Criminal class. See src/streetcrime/agents/criminal.py. Default: CriminalParams()
    """

    params: CriminalParams = CriminalParams()
    dtypes: dict[str, str] = {
        "crime_motivation": "float16",
        "successfull_crime": "bool",
        "victim": "int64",
        "close_agents": "object",
    }

    @classmethod
    def __init__(cls, params: CriminalParams = CriminalParams()):
        """Initializes the Criminal class.

        Parameters
        ----------
        params : CriminalParams
            The parameters of the Criminal. Default: CriminalParams()"""
        super().__init__(params)
        cls.model.agents.successfull_crime = False
        cls._set_crime_motivation()

    @classmethod
    def step(cls) -> None:
        """The step method of the Criminal class.
        If the Criminal is in transport, finds a suitable victim and commits a crime."""
        # super().step()
        criminal_mask = cls.mask & (cls.model.agents.status == "transport")

        for criminal in cls.model.agents[criminal_mask].itertuples():
            # Find victims
            available_victims = cls.model.agents[
                (cls.model.agents["status"] == "transport")
                & (cls.model.agents["type"].str.contains("Worker"))
                & (~cls.model.agents["targeted"])
            ]
            close_agents = (
                cls.model.agents["geometry"].distance(criminal.geometry)
                <= cls.params.opportunity_awareness
            )
            police_agents = close_agents & (
                cls.model.agents["type"].str.contains("Police")
            )
            if police_agents.any():
                continue
            close_victims = available_victims[
                close_agents.reindex(available_victims.index)
            ]
            if (
                close_victims["crime_attractiveness"].sum() == 0
            ):  # if no victims or crime attractiveness is 0 for all victims
                continue
            victim = int(
                close_victims.sample(
                    1, weights=close_victims["crime_attractiveness"]
                ).index[0]
            )  # TODO: instead of iterating over criminals, sample all at once and
            cls.model.agents.loc[victim, "targeted"] = True
            cls.model.agents.loc[criminal.Index, "victim"] = victim
            cls.model.agents.loc[criminal.Index, "close_agents"] = len(close_victims)

        ### Compute whether the crime is successfull

        committed_crimes = criminal_mask & (cls.model.agents["victim"].notna())

        if type(cls) == Pickpocket:
            cls.model.agents.loc[
                committed_crimes, "successfull_crime"
            ] = cls.model.agents.loc[
                committed_crimes, "crime_motivation"
            ] + cls.params.crowd_effect * (
                cls.model.agents.loc[committed_crimes, "close_agents"] - 1
            ) + np.random.normal(
                0, 0.05
            ) >= np.random.uniform(
                0, 1
            )
        else:
            cls.model.agents.loc[
                committed_crimes, "successfull_crime"
            ] = cls.model.agents.loc[
                committed_crimes, "crime_motivation"
            ] - cls.params.crowd_effect * (
                cls.model.agents.loc[committed_crimes, "close_agents"] - 1
            ) + np.random.normal(
                0, 0.05
            ) >= np.random.uniform(
                0, 1
            )  # TODO: missing the effect of the victim's cls defence

        cls.model.crimes = pd.concat(
            [
                cls.model.crimes,
                cls.model.agents.loc[
                    committed_crimes, ["victim", "geometry", "type", "close_agents"]
                ],
            ]
        )
        cls.model.crimes.loc[
            cls.model.crimes["datetime"].isna(), "datetime"
        ] = cls.model.datetime

        cls.model.agents.victim = np.nan
        cls.model.agents.successfull_crime = False
        cls.model.agents.close_agents = np.nan
        cls.model.agents.targeted = False

        """Checks for crime opportunities in the vicinity and commits one if conditions permit."""
        """if len(police) > 0: #Awareness of police in the vicinity
            neighborhood_id = [self.model.space.find_neighborhood_by_pos(police[0].geometry)]
            if neighborhood_id is not None:
                #TODO: the data which stores crimes should be updated in the model
                #self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']] = self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']].astype(float)
                #self.model.info_neighborhoods.at[(self.unique_id, neighborhood_id), 'yesterday_police'] += 1
                #self.model.info_neighborhoods.at[(self.unique_id, neighborhood_id), 'run_police'] += 1
                #self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']] = self.model.info_neighborhoods.loc[:, ['yesterday_police', 'run_police']].astype(pd.SparseDtype(float, np.nan))
                pass
            crime['prevented'] = True"""

    @classmethod
    def _set_crime_motivation(cls) -> float:
        """Sets the crime motivation attribute of the Criminal."""
        if cls.params.crime_motivation_rule == "uniform":
            cls.model.agents.loc[cls.mask, "crime_motivation"] = np.random.uniform(
                0, 1, cls.mask.sum()
            )

        # TODO: implement normal distribution and income distribution for crime motivation

        """elif attribute_name == "crime_motivation":
            attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.income) + np.random.normal(0, 0.10)
            while attribute_value < 0 or attribute_value > 1:
                attribute_value = 1 - self.model.space.neighborhoods['city_income_distribution'].iloc[0].cdf(self.income) + np.random.normal(0, 0.10)
            return attribute_value"""


class Pickpocket(Criminal):
    """A Pickpocket is a Criminal that pickpockets Workers, he has an advantage if there are more close workers."""

    # act_decision_rule: str = "buildings, weights = (1/df.geometry.distance(agent.geometry)) * df.yesterday_visits * df.run_visits * df.mean_income * (1/df.yesterday_police) * (1/df.run_police)"

    @classmethod
    def step(cls):
        pass


class Robber(Criminal):
    """A Robber is a Criminal that robs Workers, he has a disadvantage if there are more close workers."""

    # act_decision_rule: str = "buildings, weights = (1/df.geometry.distance(agent.geometry)) * (1/df.yesterday_visits) * (1/df.run_visits) * df.mean_income * (1/df.yesterday_police) * (1/df.run_police)"

    @classmethod
    def step(cls):
        pass
