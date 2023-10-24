import geopandas as gpd
import pandas as pd
import os.path # To get the path of the files
import matplotlib.pyplot as plt
from sklearn import linear_model, cluster, metrics
from scipy import stats
import numpy as np
import csv

def load_data():
    def _merge_params(model_params, agents_params):
        def _split_column(column, params):
            temp = params[column].apply(pd.Series)
            temp.rename(columns = lambda x : str(x) + f'_{column}', inplace = True)
            params.drop(columns = column, inplace = True)
            params = pd.concat([params, temp], axis=1)
            return params
    
        model_params = _split_column('p_agents', model_params)
        
        for column in ['Resident', 'Worker', 'PoliceAgent', 'Criminal', 'Pickpocket', 'Robber']:
            agents_params = _split_column(column, agents_params)
        agents_params.drop(columns = ['0_Pickpocket', '0_Robber'], inplace = True)
        
        params = model_params.merge(agents_params, left_index = True, right_index = True)
        
        return params
    runs = []
    for file in os.listdir("outputs/runs"):
        if file.startswith("run"):
            runs.append(pd.read_pickle(os.path.join("outputs/runs", file)))
    runs = pd.concat(runs)
    model_data = pd.concat(runs['model_data'].to_list()).set_index(runs.index)
    agents_data = pd.concat(runs['agents_data'].to_list()).set_index("unique_id")
    model_params = pd.concat(runs['model_params'].to_list()).set_index(runs.index)
    agents_params = pd.concat(runs['agents_params'].to_list()).set_index(runs.index)
    params = _merge_params(model_params, agents_params)
    gen_neighborhoods = pd.concat(model_data['info_neighborhoods'].to_list(), keys = model_data.index)
    gen_neighborhoods = gen_neighborhoods.loc[:, 0, :, :]
    real_neighborhoods = gpd.read_file('data/processed/info_neighborhoods_minecrime_all.gpkg').set_index('id') 
    real_neighborhoods.rename(columns = {'percentage' : 'real_percentage'}, inplace = True)
    gen_crimes = pd.concat(model_data['crimes'].to_list(), keys = model_data.index)
    gen_crimes['geometry'] = gen_crimes['position']
    gen_crimes = gpd.GeoDataFrame(gen_crimes)
    real_crimes = gpd.read_file('data/processed/crimes_minecrime.gpkg')
    crime_rate = pd.read_csv('data/processed/crime_rate.csv').set_index('type')
    neighborhoods = gpd.read_file('data/processed/neighborhoods.gpkg').set_index('id')
    return model_data, agents_data, model_params, agents_params, params, gen_neighborhoods, real_neighborhoods, gen_crimes, real_crimes, crime_rate, neighborhoods

def compute_info_runs(gen_crimes, params):
    info_runs = pd.DataFrame()
    info_runs['total_gen_crimes'] = gen_crimes['step'].groupby('run_id').count()
    info_runs['successful'] = gen_crimes.loc[gen_crimes['successful'], 'step'].groupby('run_id').count()
    info_runs['prevented'] = gen_crimes.loc[gen_crimes['prevented'].isnull(), 'step'].groupby('run_id').count()
    info_runs['robberies'] = gen_crimes.loc[gen_crimes['type'] == 'robbery', 'step'].groupby('run_id').count()
    info_runs['pickpocketing'] = gen_crimes.loc[gen_crimes['type'] == 'pickpocketing', 'step'].groupby('run_id').count()
    info_runs['successful_robberies'] = gen_crimes.loc[(gen_crimes['type'] == 'robbery' & gen_crimes['successful']), 'step'].groupby('run_id').count()
    info_runs['successful_pickpocketing'] = gen_crimes.loc[(gen_crimes['type'] == 'pickpocketing' & gen_crimes['successful']), 'step'].groupby('run_id').count()
    info_runs['psuccessful'] = info_runs['successful']/info_runs['total_gen_crimes']
    info_runs['pprevented'] = info_runs['prevented']/info_runs['total_gen_crimes']
    info_runs['probberies'] = info_runs['robberies']/info_runs['total_gen_crimes']
    info_runs['ppickpocketing'] = info_runs['pickpocketing']/info_runs['total_gen_crimes']
    info_runs['psuccessfulrobberies'] = info_runs['successful_robberies']/info_runs['total_gen_crimes']
    info_runs['psuccessfulpickpocketing'] = info_runs['successful_pickpocketing']/info_runs['total_gen_crimes']
    #info_runs.drop(columns = ['successful', 'robberies', 'pickpocketing', 'successful_robberies', 'successful_pickpocketing'], inplace = True)
    info_runs = info_runs.merge(params, left_index = True, right_index = True)
    info_runs['crimerate'] = info_runs['total_gen_crimes']/(info_runs['days']*info_runs['num_movers'])*365*100000 #yearly gen_crimes per 100000 inhabitants
    print("Computed info on generated crimes:")
    print(info_runs)
    return info_runs
     
def sensitivity_analysis(info_runs):
    parameters = np.log(info_runs[['PoliceAgent_p_agents', 'Pickpocket_p_agents', 'Robber_p_agents', 'p_information_Worker', 'opportunity_awareness_Criminal', 'crowd_effect_Criminal']])
    outcomes = ['crimerate', 'psuccessful', 'probberies', 'ppickpocketing']
    sensitivity_df = pd.DataFrame()
    sensitivity_df['parameter'] = pd.Series(['\% of PoliceAgents', '\% of Pickpockets', '\% of Robbers', '\% Information of Worker', 'Opportunity Awareness', 'Crowd Effect'])
    regr = linear_model.LinearRegression()
    for outcome in outcomes:
        x = parameters.loc[info_runs[outcome].notna(), :]
        y = np.log(info_runs.loc[info_runs[outcome].notna(), outcome])
        #sensitivity_df[outcome] = (regr.coef_*100).round(2)
        regr.fit(x, y)
        conf_int = _regression_conf_int(0.05, regr, x, y)
        conf_int = conf_int.iloc[1:]
        sensitivity_df[outcome + 'lower'] = np.round(conf_int['lower']*100, 2).reset_index(drop = True)
        sensitivity_df[outcome + 'upper'] = np.round(conf_int['upper']*100, 2).reset_index(drop = True)
        sensitivity_df[outcome + 'significant'] = conf_int.apply(lambda x: "$^*$" if x['lower']*x['upper']>0 else None, axis = 1).reset_index(drop = True)
    sensitivity_df = sensitivity_df.iloc[:-1]
    print("Computed sensitivity analysis")
    print(sensitivity_df)
    sensitivity_df.to_csv("outputs/tables/sensitivity_analysis.csv", quoting=csv.QUOTE_NONE)
    return sensitivity_df  

def empirical_validity(gen_neighborhoods, real_neighborhoods, 
                       gen_crimes, real_crimes, crime_rate, info_runs,):
    

    #Add metrics: deviation from real percentage
    
    gen_neighborhoods = gen_neighborhoods.merge(gen_neighborhoods['run_crimes'].groupby('run_id').sum().rename('sum_crimes'), left_on = 'run_id', right_index = True)
    gen_neighborhoods['gen_percentage'] = (gen_neighborhoods['run_crimes'] / gen_neighborhoods['sum_crimes'])*100
    gen_neighborhoods = gen_neighborhoods.merge(real_neighborhoods['real_percentage'], left_on = 'neighborhood_id', right_index = True)       
    gen_neighborhoods['deviation_h'] = abs(gen_neighborhoods['gen_percentage'] - gen_neighborhoods['real_percentage'])
    gen_neighborhoods = gen_neighborhoods['deviation_h'].reset_index().set_index(['run_id'])
    
    #Deviation for all neighborhoods
    info_runs = info_runs.merge(_mean_conf_int(gen_neighborhoods['deviation_h'].drop(columns = ['neighborhood_id'])), left_index = True, right_index = True, how = 'left')
    info_runs.rename(columns = {'lower' : 'deviation_h_lower', 'upper' : 'deviation_h_upper'}, inplace = True)
    info_runs['deviation_h_mean'] = gen_neighborhoods['deviation_h'].mean()
    
    #Deviation for hotspots
    hotspots = ["DUOMO", "STAZIONE CENTRALE - PONTE SEVESO", "BUENOS AIRES - PORTA VENEZIA - PORTA MONFORTE", "TALIEDO - MORSENCHIO - Q.RE FORLANINI"]
    hotspots_indexes = real_neighborhoods.loc[real_neighborhoods['name'].isin(hotspots), :].index 
    info_runs = info_runs.merge(_mean_conf_int(gen_neighborhoods.loc[gen_neighborhoods['neighborhood_id'].isin(hotspots_indexes)].drop(columns = ['neighborhood_id'])), left_index = True, right_index = True, how = 'left')
    info_runs.rename(columns = {'lower' : 'deviation_hotspots_lower', 'upper' : 'deviation_hotspots_upper'}, inplace = True)

    #Add metrics: deviation from real position of crimes

    def nearest_crime(crimes, position = 0):
        row = crimes[['run_id', 'level_1']]
        crimes = crimes.drop(['run_id', 'level_1']).sort_values()
        return pd.concat([row, pd.Series({'nearest_crime' : crimes.index[position], 'distance' : crimes.iloc[position]})])
    distance_crimes = gen_crimes['geometry'].apply(lambda x: x.distance(real_crimes['geometry'])).reset_index()
    nearest_crimes = distance_crimes.apply(lambda x: nearest_crime(x), axis = 1)
    duplicated_crimes = nearest_crimes[nearest_crimes.duplicated(subset = ['run_id', 'nearest_crime'])]
    position = 1
    while duplicated_crimes.empty == False:
        nearest_crimes.loc[nearest_crimes.index.isin(duplicated_crimes.index)] = distance_crimes.apply(lambda x: nearest_crime(x, position), axis = 1)
        duplicated_crimes = nearest_crimes[nearest_crimes.duplicated(subset = ['run_id', 'nearest_crime'])]
        position += 1
    nearest_crimes = nearest_crimes.drop('nearest_crime', axis = 1).set_index(['run_id'])
    info_runs = info_runs.merge(_mean_conf_int(nearest_crimes['distance']), left_index = True, right_index = True, how = 'left')
    info_runs.rename(columns = {'lower' : 'deviation_crime_lower', 'upper' : 'deviation_crime_upper'}, inplace = True)
    info_runs['deviation_crime_mean'] = info_runs[['deviation_crime_lower', 'deviation_crime_upper']].mean(axis = 1)

    #Add metrics: deviation from real crime rate
    info_runs = info_runs.assign(crime_rate_robbery = crime_rate.at['robbery', 'crime_rate'], crime_rate_pickpocketing = crime_rate.at['pickpocketing', 'crime_rate'])
    info_runs['probberies'] = info_runs['probberies'].fillna(0)
    info_runs['ppickpocketing'] = info_runs['ppickpocketing'].fillna(0)
    info_runs['deviation_crime_rate'] = info_runs['crimerate']*info_runs['probberies'] - info_runs['crime_rate_robbery'] + info_runs['crimerate']*info_runs['ppickpocketing'] - info_runs['crime_rate_pickpocketing']
     
    #Ranking
    info_runs['ranking'] = 1/2 * info_runs['deviation_h_mean'] + 1/2 * info_runs['deviation_crime_mean']
    info_runs.sort_values(by = 'ranking', inplace = True)

    #Process table for latex
    info_runs['act_decision_rule_Criminal'] = info_runs['act_decision_rule_Robber'].apply(lambda x: "Logarithmic weighted distance" if "log" in str(x) else "Linearly weighted distance")
    duplicated_columns = ['PoliceAgent_p_agents', 'Pickpocket_p_agents', 'Robber_p_agents', 'Worker_p_agents', 'crowd_effect_Criminal', 'p_information_Worker', 'act_decision_rule_Criminal']
    info_runs = info_runs.drop(index = info_runs[info_runs.duplicated(duplicated_columns)].index)
    info_runs = info_runs.drop(index = info_runs[info_runs['deviation_h_lower'] < 0].index)
    info_runs = info_runs.iloc[:10]
    info_runs = info_runs.reset_index().reset_index().rename(columns = {'index' : 'run'})
    info_runs['run'] = info_runs['run'] + 1
    percentages = ['PoliceAgent_p_agents', 'Pickpocket_p_agents', 'Robber_p_agents',
                   'Worker_p_agents', 'crowd_effect_Criminal', 'p_information_Worker']
    rounded = ['deviation_h_lower', 'deviation_h_upper', 'deviation_crime_lower', 'deviation_crime_upper']
    integers = ['deviation_crime_rate']
    _save_table(info_runs, "info_runs", rounded, percentages, integers)
    print("Computed empirical validity")
    print(info_runs)
    return info_runs

def policing_experiment(info_runs):
    info_runs['policing'] = None
    info_runs.loc[info_runs['act_decision_rule_PoliceAgent'] == '', 'policing'] = '100\% Random Policing'
    info_runs.loc[info_runs['act_decision_rule_PoliceAgent'] == 'yesterday_crimes * run_crimes', 'policing'] = '100\% Hotspot Policing'
    info_runs.loc[info_runs['act_decision_rule_PoliceAgent'] == "random.choice("", 'yesterday_crimes * run_crimes')", 'policing'] = '50\% Random Policing, 50\% Hotspot Policing' #FIX HERE
    info_runs.loc[info_runs['p_information_PoliceAgent'] == 0.5] = '100\% Hotspot Policing with a 0.5 information'
    for outcome in ['pprevented', 'psuccesfullrobberies', 'psucessfullpickpocketing']:
        info_runs = info_runs.merge(info_runs.apply(lambda x: _mean_conf_int(x.groupby('policing'))), right_index = True, left_index = True, how = 'left')
    return info_runs

def long_run(info_runs, gen_crimes, model_params):
    gen_crimes.reset_index(inplace = True)
    info_runs.reset_index(inplace = True)
    
    #Selecting runs longer than 7 days
    long_run_models = model_params[(model_params['days'] >= 7)].index
    info_long_runs = info_runs[info_runs['run_id'].isin(long_run_models)]

    #Add metrics: silhouette score (clustering)
    gen_crimes = gpd.GeoDataFrame(gen_crimes, geometry = 'position')
    info_long_runs['silhouette_score'] = _silhouette_score(gen_crimes['position'])
    
    #Add metrics: daily neighborhood variance
    info_long_runs['daily_neighborhood_variance'] = gen_crimes[['run_id', 'date', 'neighborhood']].groupby(['run_id', 'date']).value_counts().var()

    #info_long_runs = info_runs
    gen_crimes = gen_crimes[gen_crimes['run_id'].isin(long_run_models)]
    
    #First plot total crimes, robberies, pickpocketing per day
    def first_plot():
        info_long_runs = info_long_runs.merge(gen_crimes[['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_crimes'}).reset_index().set_index('run_id').fillna(0), on = 'run_id', how = 'left')
        info_long_runs = info_long_runs.merge(gen_crimes.loc[gen_crimes['type'] == 'robbery', ['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_robberies'}).fillna(0), on = ['run_id', 'date'], how = 'left')
        info_long_runs = info_long_runs.merge(gen_crimes.loc[gen_crimes['type'] == 'pickpocketing', ['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_pickpocketing'}).fillna(0), on = ['run_id', 'date'], how = 'left')
        for column, label in zip(['daily_crimes', 'daily_robberies', 'daily_pickpocketing'], ['All Crimes', 'Robberies', 'Pickpocketing']):
            info_long_runs[column + '_rate'] = info_long_runs[column]/(info_long_runs['days']*info_long_runs['num_movers'])*365*100000
            info_long_runs[column + '_rate'] = info_long_runs[column + '_rate'].fillna(0)
            info_long_runs[[column + '_rate', 'date']].groupby('date').mean()
            y = info_long_runs[[column + '_rate', 'date']].groupby('date').mean().reset_index().sort_values(by = 'date')
            plt.plot(pd.to_datetime(y['date']).dt.day, y[column + '_rate'], label = label, marker='o', linewidth=3)

    plt.gca().ticklabel_format(axis='y', style='plain')
    plt.legend(loc ='upper left')
    plt.title("Daily Evolution of Crime Rates")
    plt.xlabel('Days')
    plt.ylabel('Yearly Crime Rate per 100k Inhabitants')
    plt.savefig(f"./outputs/plots/long_run_crime_rates.pdf")
    plt.show()

    #Second plot psuccessfulcrimes, psuccessfulrobberies, psuccessfulpickpocketing per day
    def second_plot():
        info_long_runs = info_long_runs.merge(gen_crimes.loc[gen_crimes['successful'], ['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_successful_crimes'}).fillna(0), on = ['run_id', 'date'], how = 'left')
        info_long_runs = info_long_runs.merge(gen_crimes.loc[(gen_crimes['successful']) & (gen_crimes['type'] == 'robbery'), ['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_successful_robberies'}).fillna(0), on = ['run_id', 'date'], how = 'left')
        info_long_runs = info_long_runs.merge(gen_crimes.loc[(gen_crimes['successful']) & (gen_crimes['type'] == 'pickpocketing'), ['step', 'date', 'run_id']].groupby(['date', 'run_id']).count().rename(columns = {'step' : 'daily_successful_pickpocketing'}).fillna(0), on = ['run_id', 'date'], how = 'left')
        for column, label in zip(['daily_successful_crimes', 'daily_successful_robberies', 'daily_successful_pickpocketing'], ['All Crimes', 'Robberies', 'Pickpocketing']):
            info_long_runs[column + '_p'] = info_long_runs[column]/info_long_runs['daily_crimes']
            info_long_runs[column + '_p'] = info_long_runs[column + '_p'].fillna(0)
            info_long_runs[[column + '_p', 'date']].groupby('date').mean()
            y = info_long_runs[[column + '_p', 'date']].groupby('date').mean().reset_index().sort_values(by = 'date')
            plt.plot(pd.to_datetime(y['date']).dt.day, y[column + '_p'], label = label, marker='o', linewidth=3)
        plt.legend(loc ='upper left')
        plt.title("Daily Evolution of Success Rates for Crimes")
        plt.xlabel('Days')
        plt.ylabel('Success Rate')
        plt.savefig(f"./outputs/plots/long_run_success_rate.pdf")
        plt.show()
    
    #Add metrics: victim reccurence (gini index)
    info_long_runs = info_long_runs.merge(gen_crimes[['step', 'run_id', 'victim']].groupby(['run_id', 'victim']).count().reset_index().set_index('run_id').rename(columns = {'step' : 'victim_recurrence'}), on = 'run_id')
    info_long_runs = info_long_runs.merge(info_long_runs[['victim_recurrence', 'run_id']].groupby('run_id').apply(_gini).rename('gini_offender'), on = 'run_id')
    
    #Add metrics: offender reccurence (gini index)
    info_long_runs = info_long_runs.merge(gen_crimes[['step', 'run_id', 'criminal']].groupby(['run_id', 'criminal']).count().reset_index().set_index('run_id').rename(columns = {'step' : 'criminal_recurrence'}), on = 'run_id')
    info_long_runs = info_long_runs.merge(info_long_runs[['criminal_recurrence', 'run_id']].groupby('run_id').apply(_gini).rename('gini_criminal'), on = 'run_id')
    
    _save_table(info_long_runs, "info_long_runs")
    return info_long_runs

def plot_gen_crimes(gen_crimes, gen_neighborhoods, name, title):
    gen_crimes = gen_crimes.loc[gen_crimes['neighborhood'].isna() == False, :]
    gen_neighborhoods = pd.DataFrame()
    plot = gen_neighborhoods.plot(column='percentage', 
                                cmap = 'OrRd', 
                                legend=True, 
                                legend_kwds={'label': "Percentage on total gen_crimes"},
                                missing_kwds={'color': 'lightgrey'})
    gen_crimes.plot(ax = plot, color = 'black', markersize = 0.5, alpha = 0.2)
    gen_neighborhoods['coords'] = gen_neighborhoods['geometry'].apply(lambda x: x.representative_point().coords[:])
    gen_neighborhoods['coords'] = [coords[0] for coords in gen_neighborhoods['coords']]
    for idx, row in gen_neighborhoods.iterrows():
        if row['percentage'] is not None:
            plt.annotate(text=round(row['percentage'], 2), xy=row['coords'],
                    horizontalalignment='center', fontsize = 5)
    plot.set_axis_off()
    plt.title(title)
    plt.savefig(f"./outputs/plots/gen_crimes_{name}.pdf")
    plt.show()
    
def _regression_conf_int(alpha, lr, X=None, y=None):
    
    """
    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """
    
    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = X.copy()
    X_aux.insert(0, 'const', 1)
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - lr.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha/2, dof)
    gap = t_val * np.sqrt(mse * var_params)

    return pd.DataFrame({
        'lower': coefs - gap, 'upper': coefs + gap
    }, index=X_aux.columns)

def _mean_conf_int(data, alpha=0.05):
    se = data.groupby('run_id').apply(lambda x: stats.sem(x, nan_policy = 'omit'))
    mean = data.groupby('run_id').mean()
    conf_int = [stats.t.interval(1-alpha, df = len(data.loc[run_id]), loc=mean.loc[run_id], scale=se.loc[run_id]) for run_id in se[se.notnull()].index]
    return pd.DataFrame(index = se[se.notnull()].index, columns=['lower', 'upper'], data = conf_int)

def _silhouette_score(data, distance = 500):
    coordinates = data.get_coordinates().to_numpy()
    dbscan = cluster.DBSCAN(eps = distance).fit(coordinates)
    silhouette_score = metrics.silhouette_score(coordinates, dbscan.labels_)
    print(silhouette_score)
    return silhouette_score

def _gini(data):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    data = data.to_numpy()
    mad = np.abs(np.subtract.outer(data, data)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def _save_table(df, name, rounded = None, percentages = None, integers = None):
    if not percentages is None:
        df[percentages] = (df[percentages]*100).round(0)
    if not rounded is None:
        df[rounded] = df[rounded].astype(float).round(2)
    if not integers is None:
        df[integers] = df[integers].round(0).astype(int)
    df.rename(columns = lambda x: x.replace("_" , "") if "_" in x else x, inplace = True)
    df.to_csv(f"outputs/tables/{name}", quoting=csv.QUOTE_NONE)

def main():
    model_data, agents_data, model_params, agents_params, params, gen_neighborhoods, real_neighborhoods, gen_crimes, real_crimes, crime_rate, neighborhoods = load_data()
    info_runs = compute_info_runs(gen_crimes, params)
    #sensitivity_df = sensitivity_analysis(info_runs)
    empirical_df = empirical_validity(gen_neighborhoods, real_neighborhoods, gen_crimes, real_crimes, crime_rate, info_runs)
    
    long_run_df = long_run(info_runs, gen_crimes, model_params)

if __name__ == "__main__":
    main()

    

#Compute total gen_crimes, successful gen_crimes and yearly crime rate


#Compute general info on neighborhoods

#col_visits = gen_neighborhoods.columns.str.contains('visits') & gen_neighborhoods.columns.str.contains('2')
#col_police = gen_neighborhoods.columns.str.contains('police') & gen_neighborhoods.columns.str.contains('2')
#gen_neighborhoods.loc[:, col_gen_crimes].groupby(['run_id', 'neighborhood_id']).mean(axis = 1)

    

#Compute total gen_crimes, successful gen_crimes and yearly crime rate


#Compute general info on neighborhoods

#col_visits = gen_neighborhoods.columns.str.contains('visits') & gen_neighborhoods.columns.str.contains('2')
#col_police = gen_neighborhoods.columns.str.contains('police') & gen_neighborhoods.columns.str.contains('2')
#gen_neighborhoods['avg_daily_gen_crimes'] = gen_neighborhoods.loc[:, col_gen_crimes].mean(axis = 1)
#gen_neighborhoods.loc[:, col_gen_crimes].groupby(['run_id', 'neighborhood_id']).mean(axis = 1)