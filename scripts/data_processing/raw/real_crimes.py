import geopandas as gpd
import osmnx as ox
import pandas as pd
import warnings
from shapely import Point
from matplotlib import pyplot as plt

import geopandas as gpd
import osmnx as ox
import pandas as pd
import warnings
from shapely import Point
from matplotlib import pyplot as plt

def process_data():
    #Reading crimes files
    crimes_caldarini = gpd.read_file('./data/raw/ARINI-map-3 (1).geojson')
    caldarini_2 = gpd.read_file('./data/raw/ARINI-map.geojson')
    crimes_caldarini = pd.concat([crimes_caldarini, caldarini_2], ignore_index = True)
    crimes_caldarini.to_crs(epsg = 7791, inplace = True)
    crimes_caldarini.to_file("./data/processed/crimes_caldarini.gpkg")

    crimes_minecrime = pd.read_csv('./data/raw/Furti_Rapine_Q1_2018_2022.csv')
    crimes_minecrime.loc[:, 'geometry'] = crimes_minecrime.apply(lambda x: Point(x['coords.lon'], x['coords.lat']), axis = 1)
    crimes_minecrime = gpd.GeoDataFrame(crimes_minecrime, geometry = 'geometry')
    crimes_minecrime.set_crs(epsg = 4326, inplace = True)
    crimes_minecrime.to_crs(epsg = 7791, inplace = True)
    crimes_minecrime.to_file("./data/processed/crimes_minecrime.gpkg")

    crimes_minecrime['date'] = pd.to_datetime(crimes_minecrime['date'])
    crimes_minecrime['year'] = crimes_minecrime['date'].dt.year
    crimes = pd.concat([crimes_caldarini, crimes_minecrime], ignore_index = True)

    #Reading city files
    neighborhoods = gpd.read_file('./data/processed/neighborhoods.gpkg').set_index('id')
    
    #Reading crimerate files
    crime_rate = pd.read_csv('data/raw/DCCV_DELITTIPS_09092023123836519.csv')
    crime_rate.rename(columns = {'REATI_PS':'type', 'Value' : 'crime_rate'}, inplace = True)
    crime_rate = crime_rate[['type', 'crime_rate']]
    crime_rate.loc[crime_rate['type'] == 'PICKTHEF', 'type'] = "pickpocketing"
    crime_rate.loc[crime_rate['type'] == 'STREETROB', 'type'] = "robbery"
    crime_rate.loc[crime_rate['type'] == 'robbery', 'crime_rate'] += crime_rate.loc[crime_rate['type'] == 'BAGTHEF', 'crime_rate'].iat[0]
    crime_rate = crime_rate[crime_rate['type'] != 'BAGTHEF'].set_index('type')
    crime_rate.to_csv('data/processed/crime_rate.csv')
    
    print('Data processed')
    return crimes_caldarini, crimes_minecrime, crimes, neighborhoods, crime_rate

#Categorizing crimes in each neighborhood
def point_in_neighborhood(point, neighborhoods):
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
        crimes.loc[:, 'neighborhood'] = crimes.geometry.apply(lambda x: point_in_neighborhood(x, neighborhoods))
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
        info_neighborhoods.to_file(f"data/processed/info_neighborhoods_{filename}.gpkg")
    plot_crimes(crimes, info_neighborhoods, name = filename, title = plot_title)
    
crimes_caldarini, crimes_minecrime, crimes, neighborhoods, crimes_rate = process_data()

crimes_processing('caldarini', "Criminal neighborhoods according to Milano Crime Monitoring (all crimes)", crimes_caldarini, neighborhoods)
crimes_processing('minecrime_robbery', "Criminal neighborhoods according to MineCrime (robbery)", crimes_minecrime[crimes_minecrime['category'] == 'RAPINA'], neighborhoods)
crimes_processing('minecrime_theft', "Criminal neighborhoods according to MineCrime (theft)", crimes_minecrime[crimes_minecrime['category'] == 'FURTO'], neighborhoods)
crimes_processing('minecrime_all', "Criminal neighborhoods according to MineCrime (all crimes)", crimes_minecrime, neighborhoods)
crimes_processing('all', "Criminal neighborhoods according to both sources (all crimes)", crimes, neighborhoods)

var = pd.DataFrame()
for year in crimes_minecrime['year'].unique():
    info_neighborhoods = create_info_neighborhoods(crimes_minecrime[crimes_minecrime['year'] == year], neighborhoods)
    var[year] = info_neighborhoods['percentage']
info_neighborhoods['variance p'] = var.var(axis = 1)
info_neighborhoods.boxplot(column = 'variance p')
plt.savefig(f"./outputs/plots/variance_p.pdf")



    

