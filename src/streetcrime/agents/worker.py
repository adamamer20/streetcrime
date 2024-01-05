from dataclasses import dataclass

import numpy as np
import pandas as pd
from streetcrime.agents.resident import Resident, ResidentParams


@dataclass
class WorkerParams(ResidentParams):
    """
    The WorkerParams class is a dataclass that contains the parameters of the Worker class.

    Attributes:
    ----------
    mean_work_start_time : float
        The mean time when the Worker starts working. Default: 8
    sd_work_start_time : float
        The standard deviation of the time when the Worker starts working. Default: 2
    mean_work_end_time : float
        The mean time when the Worker ends working. Default: 18
    sd_work_end_time : float
        The standard deviation of the time when the Worker ends working. Default: 2
    mean_self_defence : float
        The mean probability of the Worker to defend himself when robbed. Default: 0.5
    sd_self_defence : float
        The standard deviation of the probability of the Worker to defend himself when robbed. Default: 0.17
    mean_crime_attractiveness : float
        The mean probability of the Worker to be targeted by a criminal. Default: 0.5
    sd_crime_attractiveness : float
        The standard deviation of the probability of the Worker to be targeted by a criminal. Default: 0.17
    crime_attractiveness_rule : str
        The rule used to assign the crime attractiveness attribute to the Worker. Default: "uniform"
    """

    # act_decision_rule: str = "buildings, weights = 1/df.geometry.distance(agent.geometry)*1/(df.yesterday_crimes+1) * 1/(df.run_crimes+1)"
    mean_work_start_time: float = 8
    sd_work_start_time: float = 2
    limits_work_start_time: tuple[float] = (5, 24)
    mean_work_end_time: float = 18
    sd_work_end_time: float = 2
    limits_work_end_time: tuple[float] = (0, 24)
    mean_self_defence: float = 0.5
    sd_self_defence: float = 0.3
    mean_crime_attractiveness: float = 0.5
    sd_crime_attractiveness: float = 0.3
    crime_attractiveness_rule: str = "uniform"


class Worker(Resident):
    """The Worker class is a subclass of Resident. Workers are assigned a home based on the proportion of population in each neighborhood and have a fixed work place and working hours.
    Everyday, they go to work and do activities if they don't have to work or rest. They choose activities based on recent crimes in the neighborhood.

    Attributes:
    ----------
    dtypes : dict[str, str]
        The attributes of the Agent as a dictionary of columns and data types. It contains:
        - work_start_time : float16
            -- The time when the Worker starts working
        - work_end_time : float16
            -- The time when the Worker ends working
        - work : int32
            -- The id of the node where the Worker works. Assigned randomly when the Worker is created.
        - self_defence : float16
            -- Affects the probability of the Worker to defend himself when attacked.
        - crime_attractiveness : float16
            -- Affects the probability of the Worker to be targeted by a criminal.
        - targeted : bool
            -- True if the Worker is being targeted by a criminal in current step.
    """

    params: WorkerParams = WorkerParams()
    dtypes: dict[str, str] = {
        "work_start_time": "float16",
        "work_end_time": "float16",
        "work": "int32",
        "self_defence": "float16",
        "crime_attractiveness": "float16",
        "targeted": "bool",
    }

    @classmethod
    def __init__(cls, params: WorkerParams = WorkerParams()) -> None:
        """Initializes the Worker class.

        Parameters:
        ----------
        params : WorkerParams
            The parameters of the Worker. Default: WorkerParams
        """
        super().__init__(params)
        cls._set_crime_attractiveness()
        cls._set_working_time()
        cls.model.agents.loc[cls.mask, "work"] = cls.model.space.get_random_nodes(
            function="work", n=cls.mask.sum()
        )
        cls.model.agents.loc[cls.mask, "defence"] = cls.gen_attribute(
            limits=[0, 1],
            attribute_type="float",
            distribution="normal",
            mean=cls.params.mean_self_defence,
            sd=cls.params.sd_self_defence,
            n=cls.mask.sum(),
        )

        cls.model.agents.loc[cls.mask, "crime_attractiveness"] = cls.gen_attribute(
            limits=[0, 1],
            attribute_type="float",
            distribution="normal",
            mean=cls.params.mean_crime_attractiveness,
            sd=cls.params.sd_crime_attractiveness,
            n=cls.mask.sum(),
        )
        cls.model.agents.targeted = False
        # self.crime_attractiveness = self._gen_crime_attractiveness()

    @classmethod
    def step(cls) -> None:
        """The step method of the Worker class.
        The Worker goes to work if it is working time and does activities if it is not working time.
        """
        # If it is time to go to work, set destination as work if not already there or on the way

        work_start_time = (
            pd.to_datetime(cls.model.datetime.date())
            + pd.to_timedelta(np.floor(cls.model.agents.work_start_time), unit="H")
            + pd.to_timedelta((cls.model.agents.work_start_time % 1) * 60, unit="m")
        )
        work_end_time = (
            pd.to_datetime(cls.model.datetime.date())
            + pd.to_timedelta(np.floor(cls.model.agents.work_end_time), unit="H")
            + pd.to_timedelta((cls.model.agents.work_end_time % 1) * 60, unit="m")
        )
        work_mask = (work_start_time <= cls.model.datetime) & (
            cls.model.datetime <= work_end_time
        )
        mask = (
            cls.mask
            & work_mask
            & (cls.model.agents.node != cls.model.agents.work)
            & (cls.model.agents.destination != cls.model.agents.work)
        )
        cls.model.agents.loc[mask, "destination"] = cls.model.agents.loc[mask, "work"]

        # If it is not working time and at work, set status as free
        free_mask = cls.mask & (~work_mask) & (cls.model.agents.status == "work")
        cls.model.agents.loc[free_mask, "status"] = "free"
        # super().step()

    @classmethod
    def _set_working_time(cls):
        """Sets the working time for each agent."""
        cls.model.agents.loc[cls.mask, "work_start_time"] = cls.gen_attribute(
            limits=[0, 24],
            mean=cls.params.mean_work_start_time,
            sd=cls.params.sd_work_start_time,
            n=cls.mask.sum(),
        )
        cls.model.agents.loc[cls.mask, "work_end_time"] = cls.gen_attribute(
            limits=[0, 24],
            mean=cls.params.mean_work_end_time,
            sd=cls.params.sd_work_end_time,
            n=cls.mask.sum(),
        )

    @classmethod
    def _set_crime_attractiveness(cls) -> None:
        """Sets the crime attractiveness for each agent."""
        if cls.params.crime_attractiveness_rule == "uniform":
            cls.model.agents.loc[cls.mask, "crime_attractiveness"] = np.random.uniform(
                0, 1, cls.mask.sum()
            )

        """#TODO: implement normal distribution and income distribution for crime attractiveness   
            crime_attractiveness = self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
            while crime_attractiveness < 0 or crime_attractiveness > 1:
                crime_attractiveness =  self.model.space.income_distribution.cdf(self.income) + np.random.normal(0, 0.10)
            return crime_attractiveness"""
