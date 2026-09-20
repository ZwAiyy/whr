import pandas as pd

df = pd.read_csv('data.csv',parse_dates=['date'])
df.drop(columns=['Unnamed: 0'], inplace=True)
#df.rename(columns={'Unnamed: 0': 'Num'}, inplace=True)
# df.drop(columns=['Num'], inplace=True)
df.to_csv('data.csv',index=False)
print(df.columns.tolist())
