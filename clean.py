import pandas as pd
import numpy as np

df= pd.read_csv('banking-marketing_combined.csv')

df = df[~df['poutcome'].isin(['other'])]
df =df[~df['job'].isin(['unknown'])]
df = df[~df['education'].isin(['unknown'])]
df = df[~(df['pdays'] == -1) ]
df = df[~(df['balance'] < 0)]

# Phase 2: Create new features
df['level'] = pd.cut(df['balance'], bins=[0, df['balance'].quantile(0.2), df['balance'].quantile(0.7), float('inf')], labels=['low', 'middle', 'high'])
df['level'] = df['level'].fillna('low')
print(pd.crosstab(df['level'], df['y']))

df['burden'] = np.where((df['housing'] == 'yes') | (df['loan'] == 'yes'), 1, 0)
print(pd.crosstab(df['burden'], df['y']))

df['possibility'] = np.select([
    (df['burden'] == 1) & (df['level'] == 'low'),
    (df['burden'] == 1) & (df['level'] == 'middle'),
    (df['burden'] == 1) & (df['level'] == 'high'),
    (df['burden'] == 0) & (df['level'] == 'low'),
    (df['burden'] == 0) & (df['level'] == 'middle')
], [
    'very low',
    'low',
    'moderate',
    'moderate',
    'high'
], default='high')
print(pd.crosstab(df['possibility'], df['y']))

# 2 new features
df['income'] = np.select([
    df['job'].isin(['entrepreneur', 'self-employed', 'management', 'admin.']),
    df['job'].isin(['technician', 'services']),
    df['job'].isin(['blue-collar', 'housemaid'])
], [
    'high',
    'moderate',
    'low'
], default='none')

df['risk'] = np.select([
    df['job'].isin(['entrepreneur', 'self-employed']),
    df['job'].isin(['management', 'admin', 'technician', 'service']),
    df['job'].isin(['blue-collar', 'housemaid'])
], [
    'high',
    'moderate',
    'low'
], default='dependent')
print(pd.crosstab(df['income'], df['y']))
print(pd.crosstab(df['risk'], df['y']))

df['poutcome'] = df['poutcome'].replace('unknown', 'failure')

df.drop(['day', 'month','contact'], axis=1, inplace=True)
