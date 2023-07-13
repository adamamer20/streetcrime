import pyproj #Type hinting for CRS
import geopandas as gpd
from simpledbf import Dbf5
import pandas as pd
from shapely.geometry import Polygon #Type hinting for geometry
import mesa
import mesa_geo as mg

class Building(mg.GeoAgent):
    unique_id: int  # an ID that represents the building
    model: mesa.Model
    geometry: Polygon
    crs: pyproj.CRS
    position: mesa.space.FloatCoordinate
    function: int
    home: bool
    work: bool
    day_act: bool
    night_act: bool
    entrance_pos: mesa.space.FloatCoordinate  # nearest vertex on road

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)
        self.entrance = None
        self.function = None
        self.position = None
        self.home = False
        self.day_act = False
        self.night_act = False

    def __repr__(self) -> str:
        return (
            f"unique_id={self.unique_id}, function={self.function}, "
            f"position={self.position})"
        )
    
    def __eq__(self, other):
        if isinstance(other, Building):
            return self.unique_id == other.unique_id
        else:
            return False

def data_processing(buildings_file : str, 
                          buildings_fun_file : str
                          ) -> gpd.GeoDataFrame:
    buildings_df = gpd.read_file(buildings_file)
    buildings_fun_dbf = Dbf5(buildings_fun_file)
    buildings_fun_df = buildings_fun_dbf.to_dataframe()
    buildings_fun_df.rename(columns = {'EDIFC_USO': 'function' }, inplace = True)
    buildings_fun_df = buildings_fun_df.astype({"function": int})
    buildings_df = buildings_df.merge(buildings_fun_df, on = "CLASSREF")
    buildings_df.drop("CLASSREF", axis=1, inplace=True)

    #Remove Transportation 
    buildings_df = buildings_df[(buildings_df['function'] != 6) | #6 Transport
                                ((buildings_df['function'] > 600) & (buildings_df['function'] < 605)) |
                                ((buildings_df['function']>60000) & (buildings_df['function'] < 60500))    
                                ]
    buildings_df.index.name = "unique_id"

    #Analyze function and categorize buildings
    buildings_df['home'] = False
    buildings_df['day_act'] = False
    buildings_df['night_act'] = False

    buildings_df.loc[(buildings_df['function'] == 1) |
                     (buildings_df['function'] == 101) , 'home'] = True #1 or 101 Residenziale o abitativa,
    buildings_df.loc[(buildings_df['function'] != 8) & #8 Industrial, 9 Agricultural
                     (buildings_df['function'] != 9) & 
                                ((buildings_df['function'] < 800) | (buildings_df['function'] > 905)) |
                                ((buildings_df['function'] < 80000) | (buildings_df['function'] > 80900))    
                                , 'day_act'] = True
    buildings_df.loc[(buildings_df['function'] == 1) | #3012 Ospedale, 306 Forze dell'Ordine, 307 Vigili del fuoco
                     (buildings_df['function'] == 101) |
                     (buildings_df['function'] == 3012) | 
                     (buildings_df['function'] == 306) | 
                     (buildings_df['function'] == 307),  
                     'night_act'] = True

    buildings_df.to_file(r'C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\processed\buildings.shp')
    return buildings_df
        
