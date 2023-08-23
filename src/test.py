import pandas as pd
import datetime as dt

date = dt.datetime(2020, 1, 1, 5, 30).date()
df = pd.DataFrame()
df = df.assign(**{str(date) : 1})
print(df)