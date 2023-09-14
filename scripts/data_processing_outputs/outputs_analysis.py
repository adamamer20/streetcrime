import geopandas as gpd
import pandas as pd
import os.path # To get the path of the files
from src.data_processing.crimes import crimes_processing

#Categorizing crimes in each neighborhood
def point_in_neighborhood(point):
    try:
        return neighborhoods[neighborhoods.geometry.contains(point)].index[0]
    except:
        return None

def plot_crimes(crimes, neighborhoods, name, title):
    crimes.loc[:, 'neighborhood'] = crimes.geometry.apply(lambda x: point_in_neighborhood(x))
    crimes = crimes.loc[crimes['neighborhood'].isna() == False, :]
    info_neighborhoods = pd.DataFrame()
    info_neighborhoods['crimes'] = crimes.value_counts('neighborhood')
    info_neighborhoods['crimes']
    info_neighborhoods['percentage'] = info_neighborhoods['crimes'] / info_neighborhoods['crimes'].sum()*100
    info_neighborhoods = info_neighborhoods.merge(neighborhoods[['geometry', 'name']], 
                                                  how = 'right', 
                                                  left_index = True, right_index = True)
    info_neighborhoods = gpd.GeoDataFrame(info_neighborhoods)
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

directory = os.path.dirname(__file__)
for i in range(2):
    parent_directory = os.path.split(directory)[0]
    directory = parent_directory

#Create single df with model data
runs = []
for file in os.listdir(os.path.join(directory, r"outputs\runs")):
    if file.startswith("run"):
        runs.append(pd.read_pickle(os.path.join(directory, r"outputs\runs", file)))
runs = pd.concat(runs)
model_data = pd.concat(runs['model_data'].to_list()).set_index(runs.index)
model_params = pd.concat(runs['model_params'].to_list()).set_index(runs.index)
agents_data = pd.concat(runs['agents_data'].to_list()).set_index("unique_id")
agents_params = pd.concat(runs['agents_params'].to_list()).set_index(runs.index)
info_neighborhoods = pd.concat(model_data['info_neighborhoods'].to_list(), keys = model_data.index)
crimes = pd.concat(model_data['crimes'].to_list(), keys = model_data.index)
crimes['geometry'] = crimes['position']
neighborhoods = gpd.read_file('./data/processed/neighborhoods.gpkg').set_index('id')

crimes_processing('simulated', "Criminal neighborhoods according to simulated data (robbery and theft)", crimes, neighborhoods)

#TESTING
#plot = agents_data['crime_motivation'].plot.box()
#fig = plot.get_figure()
#fig.savefig(os.path.join(directory, r"outputs\runs\crime_motivation.png"))
#plot2 = agents_data['crime_attractiveness'].plot.box()
#fig2 = plot2.get_figure()
#fig2.savefig(os.path.join(directory, r"outputs\runs\crime_attractiveness.png"))
info_crimes = pd.DataFrame()

#Compute total crimes, successful crimes and yearly crime rate
info_crimes['total_crimes'] = crimes['step'].groupby('run_id').count()
info_crimes['successful_crimes'] = crimes.loc[crimes['successful'], 'step'].groupby('run_id').count()
info_crimes['p_successful'] = info_crimes['successful_crimes']/info_crimes['total_crimes']
info_crimes.drop(columns = 'successful_crimes', inplace = True)
info_crimes = info_crimes.merge(model_params[['num_movers', 'days', 'len_step']], left_index=True, right_index=True)
info_crimes['crime_rate'] = info_crimes['total_crimes']/(info_crimes['days']*info_crimes['num_movers'])*365*100000 #yearly crimes per 100000 inhabitants


#Compute general info on neighborhoods
model_info_neighborhoods = info_neighborhoods.loc[:, 0, :, :]
col_crimes = model_info_neighborhoods.columns.str.contains('crimes') & model_info_neighborhoods.columns.str.contains('2')
col_visits = model_info_neighborhoods.columns.str.contains('visits') & model_info_neighborhoods.columns.str.contains('2')
col_police = model_info_neighborhoods.columns.str.contains('police') & model_info_neighborhoods.columns.str.contains('2')
model_info_neighborhoods['avg_daily_crimes'] = model_info_neighborhoods.loc[:, col_crimes].mean(axis = 1)
info_neighborhoods.loc[:, col_crimes].groupby(['run_id', 'neighborhood_id']).mean(axis = 1)