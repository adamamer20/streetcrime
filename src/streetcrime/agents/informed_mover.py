from streetcrime.agents.mover import Mover, MoverParams
from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from datetime import datetime


@dataclass
class InformedMoverParams(MoverParams):
    """The InformedMoverParams class is a dataclass that contains the parameters of an InformedMover.

    Attributes:
    ----------
    p_information : float = 1
        It defines the percentage of information the InformedMover knows about the previous day.
    """

    p_information: float = 1


class InformedMover(Mover):
    """The InformedMover class is a subclass of Mover. It can update its information about the city, grouped by neighborhood.

    Attributes:
    ----------
    p_information : float = 1
        It defines the percentage of information the InformedMover knows about the previous day.
    """
    params: InformedMoverParams = InformedMoverParams()
    dtypes: dict[str, str] = {
        # TODO: add data types
    }

    @classmethod
    def __init__(cls, params: InformedMoverParams = InformedMoverParams()) -> None:
        """Initializes the InformedMover class.
        Parameters:
        ----------
        params : InformedMoverParams
            The parameters of the InformedMover. Default: InformedMoverParams"""
        super().__init__(params)

    ''''#TODO: refactor as dataframe
    def update_info(self, info_type : str = None) -> None:
        """Gets a random sample of `p = Mover.p_information` of the previous day information from the `Mover.model.data[f'{info_type}']` dataframe, 
        groups it by neighborhood and updates the `Mover.data['info_neighborhoods']` dataframe.
     
        Parameters:
        ----------
        info_type : str 
            can be "crimes", "visits"
            
        Returns:
        ----------
        None
        
        """
        #If p_information = 1, the data is the same as the model data
        if self.p_information  == 1:
            pass
        else:
            
            yesterday = self.model.datetime.replace(day = self.model.datetime.day - 1)
            yesterday = yesterday.date()
            try:
                yesterday_data = self.model.data[f'{info_type}'][self.model.data[f'{info_type}']['date'] == yesterday]
                known_data = yesterday_data.sample(frac = self.p_information)
                data_per_neighborhood = known_data['neighborhood'].value_counts()
                for neighborhood, data in data_per_neighborhood.items():
                    self.model.info_neighborhoods.loc[self.unique_id, neighborhood, f'yesterday_{info_type}'] = data
                    self.model.info_neighborhoods.loc[self.unique_id, neighborhood, f'run_{info_type}'] += data
            except KeyError:
                self.model.info_neighborhoods = self.model.info_neighborhoods.astype(float)
                complete_info = self.model.info_neighborhoods.xs(0)
                self.model.info_neighborhoods.loc[self.unique_id].update(complete_info)
                self.model.info_neighborhoods = self.model.info_neighborhoods.astype(pd.SparseDtype(float, np.nan))'''
