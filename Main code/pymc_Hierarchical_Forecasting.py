import pandas as pd
from timeseers import LinearTrend, FourierSeasonality
import Data_Parsing as dp

path = '..\\Data'
train_filename = 'train_short'

df_input = dp.parse_csv(filename=train_filename)
_, df = dp.normalize_data(df_input)
df.rename({'timestamp':'t', 'Close':'value'}, axis=1, inplace=True)
df['t'] = pd.to_datetime(df['t'])
df = df[['t', 'value']]

model = LinearTrend(n_changepoints=10) * FourierSeasonality(n=5, period=pd.Timedelta(days=30.5))
model.fit(df[['t']], df['value'], tune=2000, chains=1, cores=1)

model.plot_components(X_true=df, y_true=df['value'])

