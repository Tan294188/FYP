import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

df= pd.read_csv('banking-marketing_combined.csv')

df = df[~df['poutcome'].isin(['other'])]
df =df[~df['job'].isin(['unknown'])]
df = df[~df['education'].isin(['unknown'])]
df = df[~(df['pdays'] == -1) ]
df = df[~(df['balance'] < 0)]

#phase 1
MinMaxScaler= MinMaxScaler()
df['previous'] = MinMaxScaler.fit_transform(df[['previous']])
df['age'] = MinMaxScaler.fit_transform(df[['age']])
df['campaign'] =MinMaxScaler.fit_transform(df[['campaign']])

StandardScaler = StandardScaler()
df['balance'] = StandardScaler.fit_transform(df[['balance']])

#phase 2: Create new features
df['level'] = pd.cut(
    df['balance'],
    bins=[df['balance'].min(), df['balance'].quantile(0.2), df['balance'].quantile(0.7), float('inf')],
    labels=['low', 'middle', 'high']
)
df['level'] = df['level'].fillna('low')

df['burden'] = np.where((df['housing'] == 'yes') | (df['loan'] == 'yes'), 1, 0)

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

df.drop(['burden','possibility'], axis=1, inplace=True)

# phase 3
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

df.drop(['income'], axis=1, inplace=True)

df['poutcome'] = df['poutcome'].replace('unknown', 'failure')

df.drop(['day', 'month','contact'], axis=1, inplace=True)


categorical_cols = df.select_dtypes(include=['object','category']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

models = KNeighborsClassifier(n_neighbors=10,weights = 'distance', metric = 'manhanttan')
X = df.drop('y', axis=1)  
y = df['y']
smote = SMOTE(random_state=42)
F1 = {}
AUC = {}
results = {}
for i in range(1,5,1):
    print(f"Test size: {i/10:.1f}, F1, AUC, Accuracy")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=i/10, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    models.fit(X_resampled, y_resampled)
    y_pred = models.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc= roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN,{f1:.3f},{auc:.3f},{accuracy:.3f}")
print("")
