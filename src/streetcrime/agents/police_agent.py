from dataclasses import dataclass

from streetcrime.agents.informed_mover import InformedMover, InformedMoverParams


@dataclass
class PoliceAgentParams(InformedMoverParams):
    # act_decision_rule = "yesterday_crimes * run_crimes"
    pass


class PoliceAgent(InformedMover):
    """
    The PoliceAgent class is a subclass of InformedMover. They patrol the city and watch for crimes.
    """

    params: PoliceAgentParams = PoliceAgentParams()

    @classmethod
    def __init__(cls, params: PoliceAgentParams = PoliceAgentParams()) -> None:
        """
        Initializes the PoliceAgent class.

        Parameters:
        ----------
        params : PoliceAgentParams
            The parameters of the PoliceAgent. Default: PoliceAgentParams
        """
        super().__init__(params)

    @classmethod
    def step(cls):
        pass
