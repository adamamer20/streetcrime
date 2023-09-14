import geopandas as gpd
import pandas as pd
import csv 


orig_neighborhoods = gpd.read_file('./data/processed/neighborhoods.gpkg')
neighborhoods = orig_neighborhoods[['name', 'prop', 'ae', 'loce', 'scalee']]
neighborhoods['prop'] = neighborhoods['prop']*100
neighborhoods['prop'] = neighborhoods['prop'].round(2)
neighborhoods.loc[:, ['ae', 'loce', 'scalee']] = neighborhoods[['ae', 'loce', 'scalee']].round().astype(int)
neighborhoods.sort_values('name', inplace = True)
city = pd.DataFrame([{'name' : 'CITY', 
                    'prop' : 100, 
                    'ae' : orig_neighborhoods.at[1, 'city_ae'], 
                    'loce' : orig_neighborhoods.at[1, 'city_loce'], 
                    'scalee' : orig_neighborhoods.at[1, 'city_scalee']}])
neighborhoods = pd.concat([neighborhoods, city], ignore_index = True)

neighborhoods.to_csv('./outputs/tables/neighborhoods.csv', index = False)