import geopandas as gpd
import pandas as pd 
from shapely import wkt

Milan_df_agents = pd.read_csv(r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\outputs\run_agents.csv")
Milan_df_agents["geometry"] = gpd.GeoSeries.from_wkt(Milan_df_agents["geometry"])
Milan_df_agents ["original_path"]= gpd.GeoSeries.from_wkt(Milan_df_agents["original_path"])
Milan_df_agents = gpd.GeoDataFrame(Milan_df_agents, geometry="geometry")
Milan_df_agents = Milan_df_agents.groupby('Step')
for i in range(20):
    color = ["blue" if x == "home" else "green" if x == "work" else "red" for x in Milan_df_agents.get_group(i)["status"]]
    base = Milan_df_agents.get_group(i).plot(zorder = 2, color = color)
    Milan_df_agents = Milan_df_agents.set_geometry("original_path")
    Milan_df_agents.get_group(i).plot(ax=base, color='grey', zorder = 1)

#Draw the points of agents for every step
