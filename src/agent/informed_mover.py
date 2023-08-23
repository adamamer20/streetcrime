import pandas as pd
from datetime import datetime, timedelta
from src.agent.mover import Mover
import mesa
from shapely.geometry import Point
import pyproj

class InformedMover(Mover):
    """The InformedMover class is a subclass of Mover. It can update its information about the city, grouped by neighborhood.

    Parameters:
    ----------
        unique_id : int
            The unique id of the Mover.
        model : mesa.Model 
            The model of the simulation where the Mover is used. See src/model/model.py
        geometry : shapely.geometry.Point
            The point geometry of the Mover in the city.
        crs : pyproj.CRS 
            The crs of the Mover (usually same as mesa_geo.GeoSpace).
    
    Attributes:
    ----------
    attributes : dict[str : pd.DataFrame]
        It defines which additional attributes a InformedMover class has with respect to its parent class.
        It can be a value or a method. The method is used for the initialization of the attribute in InformedMover.data. 
        It contains:
        - info_neighborhoods : pd.Dataframe()
            -- The dataframe containing the information known to the InformedMover in each neighborhood (columns = yesterday_{info_type}, run_{info_type})
    
    params : dict[str, float]
        It contains fixed attributes or information on how the previously specified attributes are going to be generated.
        - p_information : float
            -- The p of information (randomly sampled) the InformedMover will get about the previous day. Default: 1
            
    Methods:
    -------
    update_info(info_type : str = None) -> None
        -- Updates the information about the city of the previous day.
        
    See Also
    --------
    Mover: src/agent/mover.py
    GeoAgent: mesa/geo_agent.py
    StreetCrimeModel: src/model/model.py        
    """
    attributes : dict[str or int or list(int) or float] = {
    'info_neighborhoods' : "model.data['info_neighborhoods'].copy()"}
    
    params : dict[str, float] = {
        "p_information" : 1
    }

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
        #Only afther the first day, at midnight, update information
        if (self.model.data['datetime'].day > 1 
            and self.model.data['datetime'].hour == 0 
            and self.model.data['datetime'].minute == 0):
            #If p_information = 1, the data is the same as the model data
            if self.params['p_information'] == 1:
                self.data['info_neighborhoods'] = self.model.data['info_neighborhoods']
            else:
                yesterday = self.model.data['datetime'].replace(day = self.model.data['datetime'].day - 1)
                yesterday = yesterday.date()
                yesterday_data = self.model.data[f'{info_type}'][self.model.data[f'{info_type}']['date'] == yesterday]
                known_data = yesterday_data.sample(frac = self.params['p_information'])
                data_per_neighborhood = known_data['neighborhood'].value_counts()
                for neighborhood, data in data_per_neighborhood.items():
                    self.data['info_neighborhoods'].at[neighborhood, f'yesterday_{info_type}'] = data
                    self.data['info_neighborhoods'].at[neighborhood, f'run_{info_type}'] += data

    
    