import geopandas as gpd
import pandas as pd
import os.path # To get the path of the files
import matplotlib.pyplot as plt
from sklearn import linear_model, cluster
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
    info_runs['successful_gen_crimes'] = gen_crimes.loc[gen_crimes['successful'], 'step'].groupby('run_id').count()
    info_runs['robberies'] = gen_crimes.loc[gen_crimes['type'] == 'robbery', 'step'].groupby('run_id').count()
    info_runs['pickpocketing'] = gen_crimes.loc[gen_crimes['type'] == 'pickpocketing', 'step'].groupby('run_id').count()
    info_runs['psuccessful'] = info_runs['successful_gen_crimes']/info_runs['total_gen_crimes']
    info_runs['probberies'] = info_runs['robberies']/info_runs['total_gen_crimes']
    info_runs['ppickpocketing'] = info_runs['pickpocketing']/info_runs['total_gen_crimes']
    info_runs.drop(columns = ['successful_gen_crimes', 'robberies', 'pickpocketing'], inplace = True)
    info_runs = info_runs.merge(params, left_index = True, right_index = True)
    info_runs['crimerate'] = info_runs['total_gen_crimes']/(info_runs['days']*info_runs['num_movers'])*365*100000 #yearly gen_crimes per 100000 inhabitants
    print("Computed info gen_crimes:")
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
                       gen_crimes, real_crimes, crime_rate, info_runs):
    
    #Add performance: deviation from real percentage
    
    gen_neighborhoods = gen_neighborhoods.merge(gen_neighborhoods['run_crimes'].groupby('run_id').sum().rename('sum_crimes'), left_on = 'run_id', right_index = True)
    gen_neighborhoods['gen_percentage'] = (gen_neighborhoods['run_crimes'] / gen_neighborhoods['sum_crimes'])*100
    gen_neighborhoods = gen_neighborhoods.merge(real_neighborhoods['real_percentage'], left_on = 'neighborhood_id', right_index = True)       
    gen_neighborhoods['deviation_h'] = abs(gen_neighborhoods['gen_percentage'] - gen_neighborhoods['real_percentage'])
    info_runs = info_runs.merge(_mean_conf_int(gen_neighborhoods['deviation_h'].reset_index().set_index(['run_id']).drop(columns = ['neighborhood_id'])), left_index = True, right_index = True)
    info_runs.rename(columns = {'lower' : 'deviation_h_lower', 'upper' : 'deviation_h_upper'}, inplace = True)
    info_runs['deviation_h_mean'] = gen_neighborhoods['deviation_h'].mean()
    
    #Add performance: deviation from real position of crimes

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
    info_runs = info_runs.merge(_mean_conf_int(nearest_crimes['distance']), left_index = True, right_index = True)
    info_runs.rename(columns = {'lower' : 'deviation_crime_lower', 'upper' : 'deviation_crime_upper'}, inplace = True)
    info_runs['deviation_crime_mean'] = info_runs[['deviation_crime_lower', 'deviation_crime_upper']].mean(axis = 1)

    #Add performance: deviation from real crime rate
    info_runs = info_runs.assign(crime_rate_robbery = crime_rate.at['robbery', 'crime_rate'], crime_rate_pickpocketing = crime_rate.at['pickpocketing', 'crime_rate'])
    info_runs['probberies'] = info_runs['probberies'].fillna(0)
    info_runs['ppickpocketing'] = info_runs['ppickpocketing'].fillna(0)
    info_runs['deviation_crime_rate'] = info_runs['crimerate']*info_runs['probberies'] - info_runs['crime_rate_robbery'] + info_runs['crimerate']*info_runs['ppickpocketing'] - info_runs['crime_rate_pickpocketing']
    
    info_runs = info_runs[['PoliceAgent_p_agents', 'Pickpocket_p_agents', 'Robber_p_agents', 'Worker_p_agents', 'crowd_effect_Criminal', 'p_information_Criminal', 'p_information_Worker',
               'act_decision_rule_Pickpocket', 'act_decision_rule_Robber', 'crimerate', 'deviation_h_lower', 'deviation_h_upper', 'deviation_h_mean', 'deviation_crime_lower',
               'deviation_crime_upper', 'deviation_crime_mean', 'crime_rate_robbery','crime_rate_pickpocketing', 'deviation_crime_rate']]
    
    info_runs['ranking'] = 1/2 * info_runs['deviation_h_mean'] + 1/2 * info_runs['deviation_crime_mean']
    info_runs.sort_values(by = 'ranking', inplace = True)
    info_runs.rename(columns = lambda x: x.replace("_" , "") if "_" in x else x, inplace = True)
    percentages = ['PoliceAgentpagents', 'Pickpocketpagents', 'Robberpagents',
                   'Workerpagents', 'crowdeffectCriminal', 'pinformationWorker']
    rounded_float = ['deviationhlower', 'deviationhupper', 'deviationcrimelower', 'deviationcrimeupper']
    info_runs[percentages] = (info_runs[percentages]*100).round(0)
    info_runs[rounded_float] = info_runs[rounded_float].astype(float).round(2)
    info_runs['deviationcrimerate'] = info_runs['deviationcrimerate'].round(0).astype(int)
    info_runs['actdecisionruleCriminal'] = info_runs['actdecisionruleRobber'].apply(lambda x: "Logarithmic weighted distance" if "log" in str(x) else "Linearly weighted distance")
    duplicated_columns = ['PoliceAgentpagents', 'Pickpocketpagents', 'Robberpagents', 'Workerpagents', 'crowdeffectCriminal', 'pinformationWorker', 'actdecisionruleCriminal']
    info_runs = info_runs.drop(index = info_runs[info_runs.duplicated(duplicated_columns)].index)
    info_runs = info_runs.drop(index = info_runs[info_runs['deviationhlower'] < 0].index)
    info_runs = info_runs.iloc[:10]
    info_runs = info_runs.reset_index().reset_index().rename(columns = {'index' : 'run'})
    info_runs['run'] = info_runs['run'] + 1
    info_runs.to_csv("outputs/tables/empirical_validity.csv", quoting=csv.QUOTE_NONE)
    print("Computed empirical validity")
    print(info_runs)
    return info_runs
    
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

def _clustering(data, distance = 300):
    coordinates = arraydata.get_coordinates()
    kmeans.fit(data)
    return kmeans.labels_
model_data, agents_data, model_params, agents_params, params, gen_neighborhoods, real_neighborhoods, gen_crimes, real_crimes, crime_rate, neighborhoods = load_data()
info_runs = compute_info_runs(gen_crimes, params)
#sensitivity_df = sensitivity_analysis(info_runs)
empirical_df = empirical_validity(gen_neighborhoods, real_neighborhoods, gen_crimes, real_crimes, crime_rate, info_runs)


    

#Compute total gen_crimes, successful gen_crimes and yearly crime rate


#Compute general info on neighborhoods

#col_gen_crimes = model_gen_neighborhoods.columns.str.contains('gen_crimes') & model_gen_neighborhoods.columns.str.contains('2')
#col_visits = model_gen_neighborhoods.columns.str.contains('visits') & model_gen_neighborhoods.columns.str.contains('2')
#col_police = model_gen_neighborhoods.columns.str.contains('police') & model_gen_neighborhoods.columns.str.contains('2')
#model_gen_neighborhoods['avg_daily_gen_crimes'] = model_gen_neighborhoods.loc[:, col_gen_crimes].mean(axis = 1)
#gen_neighborhoods.loc[:, col_gen_crimes].groupby(['run_id', 'neighborhood_id']).mean(axis = 1)