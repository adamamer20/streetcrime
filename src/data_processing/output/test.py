import geopandas as gpd

def construct_distribution(loc, scale, a):
    return r"Skewed $\mathcal{N}" + fr"(\xi^e = {round(loc)}, \omega^e = {round(scale)}, \alpha^e = {round(a)})$"

neighborhoods = gpd.read_file('./data/processed/neighborhoods.gpkg')
neighborhoods.loc[:, 'Income Distribution'] = neighborhoods.apply(lambda x: construct_distribution(x.loce, x.scalee, x.ae), axis = 1)
neighborhoods = neighborhoods[['name', 'prop']]
neighborhoods.rename(columns = {'name' : 'Neighborhood', 'prop' : 'Proportion of Population'}, inplace = True)
neighborhoods.to_csv('./outputs/tables/neighborhoods.csv', index = False)
