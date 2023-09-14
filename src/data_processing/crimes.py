import geopandas as gpd
import osmnx as ox
import pandas as pd
import warnings
from shapely import Point
from matplotlib import pyplot as plt

#Categorizing crimes in each neighborhood
def point_in_neighborhood(point):
    try:
        return neighborhoods[neighborhoods.geometry.contains(point)].index[0]
    except:
        return None

def plot_crimes(crimes, info_neighborhoods, name, title):
    plot = info_neighborhoods.plot(column='percentage', 
                                cmap = 'OrRd', 
                                legend=True, 
                                legend_kwds={'label': "Percentage on total crimes"},
                                missing_kwds={'color': 'lightgrey'})
    crimes.plot(ax = plot, color = 'black', markersize = 0.5, alpha = 0.2)
    info_neighborhoods['coords'] = info_neighborhoods['geometry'].apply(lambda x: x.representative_point().coords[:])
    info_neighborhoods['coords'] = [coords[0] for coords in info_neighborhoods['coords']]
    for idx, row in info_neighborhoods.iterrows():
        if row['percentage'] is not None:
            plt.annotate(text=round(row['percentage'], 2), xy=row['coords'],
                    horizontalalignment='center', fontsize = 5)
    plot.set_axis_off()
    plt.title(title)
    plt.savefig(f"./outputs/plots/crimes_{name}.pdf")
    plt.show()
    
def create_info_neighborhoods(crimes, neighborhoods):
    if 'neighborhood' not in crimes.columns:
        crimes.loc[:, 'neighborhood'] = crimes.geometry.apply(lambda x: point_in_neighborhood(x))
        crimes = crimes.loc[crimes['neighborhood'].isna() == False, :]
    info_neighborhoods = pd.DataFrame()
    info_neighborhoods['crimes'] = crimes.value_counts('neighborhood')
    info_neighborhoods['percentage'] = info_neighborhoods['crimes'] / info_neighborhoods['crimes'].sum()*100
    info_neighborhoods = info_neighborhoods.merge(neighborhoods[['geometry', 'name', 'mean income']], 
                                                  how = 'right', 
                                                  left_index = True, right_index = True)
    corr = info_neighborhoods[['mean income', 'percentage']].corr()
    print(corr)
    info_neighborhoods['correlation income - p_crimes'] = corr.iloc[0, 1]
    info_neighborhoods = gpd.GeoDataFrame(info_neighborhoods)
    return info_neighborhoods

def crimes_processing(filename : str,
                      plot_title : str,
                      crimes: gpd.GeoDataFrame, 
                      neighborhoods: gpd.GeoDataFrame,
                      info_neighborhoods: pd.DataFrame = None):
    """
    Parameters:
    ----------
    crimes : gpd.GeoDataFrame
        The crimes dataset.
    neighborhoods : gpd.GeoDataFrame
        The neighborhoods dataset.
    """
    if info_neighborhoods is None:
        info_neighborhoods = create_info_neighborhoods(crimes, neighborhoods)
        info_neighborhoods.to_csv(f"./outputs/tables/info_neighborhoods_{filename}.csv", index = False)
    plot_crimes(crimes, info_neighborhoods, name = filename, title = plot_title)
    


    

