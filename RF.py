import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

df= pd.read_csv('banking-marketing_combined.csv')

df = df[~df['poutcome'].isin(['other'])]
df =df[~df['job'].isin(['unknown'])]
df = df[~df['education'].isin(['unknown'])]
df = df[~(df['pdays'] == -1) ]
df = df[~(df['balance'] < 0)]

#Phase 1
kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
df['age'] = kbin.fit_transform(df[['age']])
kbin2 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
df['campaign'] = kbin2.fit_transform(df[['campaign']])

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

df.drop(['level','burden','possibility'], axis=1, inplace=True)

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

df.drop(['income', 'risk'], axis=1, inplace=True)

df['poutcome'] = df['poutcome'].replace('unknown', 'failure')

df.drop(['day', 'month','contact'], axis=1, inplace=True)

categorical_cols = df.select_dtypes(include=['object','category']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

models =  RandomForestClassifier(n_estimators=85, random_state=42, max_depth=None, bootstrap=False)
X = df.drop('y', axis=1)  
y = df['y']
smote = SMOTE(random_state=42)

for i in range(1,5,1):
    print(f"Test size: {i/10:.1f}, F1, AUC, Accuracy")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=i/10, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    models.fit(X_resampled, y_resampled)
    y_pred = models.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    AUC= roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"RF,{f1:.3f},{AUC:.3f},{accuracy:.3f}")
print("")
    