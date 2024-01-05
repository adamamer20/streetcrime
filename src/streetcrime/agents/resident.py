from dataclasses import dataclass

import numpy as np
import pandas as pd

from streetcrime.agents.informed_mover import InformedMover, InformedMoverParams


@dataclass
class ResidentParams(InformedMoverParams):
    """The ResidentParams class is a dataclass that contains the parameters of the Resident class.

    Attributes
    ----------
    mean_resting_start_time : float
        The mean time when the Resident starts resting. Default: 21
    sd_resting_start_time : float
        The standard deviation of the time when the Resident starts resting. Default: 2
    mean_resting_end_time : float
        The mean time when the Resident ends resting. Default: 8
    sd_resting_end_time : float
        The standard deviation of the time when the Resident ends resting. Default: 0.83
    car_income_threshold : float
        *FUTURE IMPLEMENTATION* The income threshold above which the Resident can afford a car. Default: 3000
    home_decision_rule : str
        *FUTURE IMPLEMENTATION* The rule used to assign a home to the Resident. Default: "neighborhoods, weights = df.prop"
    """

    mean_resting_start_time: float = 21
    sd_resting_start_time: float = 2
    mean_resting_end_time: float = 8
    sd_resting_end_time: float = 0.83
    # car_income_threshold: float = 30000
    # home_decision_rule: str = "neighborhoods, weights = df.prop"


class Resident(InformedMover):
    """The Resident Class is a subclass of InformedMover. With respect to InformedMover, it has a home generates a resting timeframe every day.

    Attributes
    ----------
    dtypes : dict[str, str]
        The attributes of the Agent as a dictionary of columns and data types. It contains:
        - resting_start_time : datetime64[s]
            -- The time when the Resident starts resting.
        - resting_end_time : datetime64[s]
            -- The time when the Resident ends resting.
        - home : int32
            -- The id of the building where the Resident lives. Assigned randomly when the Resident is created.
        - income : float32
            -- *FUTURE IMPLEMENTATION*. The income of the Resident. Assigned randomly when the Resident is created.

    params : ResidentParams
        It contains fixed attributes or information on how attributes are going to be generated.

    """

    params: ResidentParams = ResidentParams()
    dtypes: dict[str, str] = {
        "resting_start_time": "datetime64[s]",
        "resting_end_time": "datetime64[s]",
        "home": "int32",
        "income": "float32",
    }

    @classmethod
    def __init__(cls, params: ResidentParams = ResidentParams()) -> None:
        """Sets Resident status and position at home
        Parameters
        ----------
        params : ResidentParams
            The parameters of the Resident. Default: ResidentParams
        """
        super().__init__(params)
        cls.set_resting_time(first_day=True)
        cls.model.agents.loc[cls.mask, "home"] = cls.model.space.get_random_nodes(
            function="home", n=cls.mask.sum()
        )
        cls.model.agents.loc[cls.mask, "status"] = "home"
        cls.model.agents.loc[cls.mask, "node"] = cls.model.agents.loc[cls.mask, "home"]
        """
        self.income = self.gen_attribute(limits=None,
                                    attribute_type='float',
                                    distribution = 'skewnorm',
                                    a = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'ae'],
                                    loc = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'loce'],
                                    scale = self.model.space.neighborhoods.at[self.model.space.buildings.at[self.home, 'neighborhood'], 'scalee'])
        self.car = self.income/self.car_income_threshold+np.random.normal(0, 0.1) > 0.5
        """

    @classmethod
    def step(cls) -> None:
        """The step method of the Resident class."""
        # If it is 14, set resting time for each agent
        if cls.model.datetime.hour == 14 and cls.model.datetime.minute == 0:
            cls.set_resting_time()

        # If it is resting time, set destination as home if not already there
        resting_mask = (cls.model.agents.resting_start_time <= cls.model.datetime) & (
            cls.model.datetime <= cls.model.agents.resting_end_time
        )
        mask = (
            cls.mask
            & resting_mask
            & (cls.model.agents["node"] != cls.model.agents["home"])
            & (cls.model.agents["destination"] != cls.model.agents["home"])
        )
        cls.model.agents.loc[mask, "destination"] = cls.model.agents.loc[mask, "home"]

        # If it is not resting time and at home, set status as free
        mask = cls.mask & (~resting_mask) & (cls.model.agents.status == "home")
        cls.model.agents.loc[mask, "status"] = "free"
        # super().step()

    @classmethod
    def set_resting_time(cls, first_day=False):
        """
        Sets the resting time for each agent.

        Parameters
        ----------
        first_day : bool
            Whether it is the first day of the simulation or not. When it's the first day, only resting_end_time is computed. Default: False
        """
        # If it's the first day, resting_start_time is model.start_datetime and we only need to compute resting_end_time
        if first_day:
            resting_end_time = cls.gen_attribute(
                limits=[0, 24],
                mean=cls.params.mean_resting_end_time,
                sd=cls.params.sd_resting_end_time,
                n=cls.mask.sum(),
            )
            cls.model.agents.loc[cls.mask, "resting_start_time"] = pd.to_datetime(
                cls.model.datetime
            )
            cls.model.agents.loc[cls.mask, "resting_end_time"] = (
                pd.to_datetime(cls.model.datetime.date())
                + pd.to_timedelta(np.floor(resting_end_time), unit="H")
                + pd.to_timedelta((resting_end_time % 1) * 60, unit="m")
            )
        # If it's not the first day, we care about both resting start and end time
        else:
            resting_start_time = cls.gen_attribute(
                limits=[0, 24],
                mean=cls.params.mean_resting_start_time,
                sd=cls.params.sd_resting_start_time,
                n=cls.mask.sum(),
            )
            resting_end_time = cls.gen_attribute(
                limits=[0, 24],
                mean=cls.params.mean_resting_end_time,
                sd=cls.params.sd_resting_end_time,
                n=cls.mask.sum(),
            )
            cls.model.agents.loc[cls.mask, "resting_start_time"] = (
                pd.to_datetime(cls.model.datetime.date())
                + pd.to_timedelta(np.floor(resting_start_time), unit="H")
                + pd.to_timedelta((resting_start_time % 1) * 60, unit="m")
            )

            cls.model.agents.loc[cls.mask, "resting_end_time"] = (
                pd.to_datetime(cls.model.datetime.date())
                + pd.to_timedelta(np.floor(resting_end_time), unit="H")
                + pd.to_timedelta((resting_end_time % 1) * 60, unit="m")
                + pd.to_timedelta(1, unit="d")
            )
