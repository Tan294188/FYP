import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

df= pd.read_csv('banking-marketing_combined.csv')

df = df[~df['poutcome'].isin(['other'])]
df =df[~df['job'].isin(['unknown'])]
df = df[~df['education'].isin(['unknown'])]
df = df[~(df['pdays'] == -1) ]
df = df[~(df['balance'] < 0)]

#Phase 1
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
df['age']= kbins.fit_transform(df[['age']])

#phase 2: Create new features
# bit reject
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

# Phase 3
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

df.drop([ 'income', 'risk'], axis=1, inplace=True)

df['poutcome'] = df['poutcome'].replace('unknown', 'failure')

df.drop(['day', 'month','contact'], axis=1, inplace=True)

# ['age'], ['duration'], ['campaign'], ['pdays'], ['previous'], ['balance']
 

categorical_cols = df.select_dtypes(include=['object','category']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

models = DecisionTreeClassifier(random_state=42,max_depth=4) 
X = df.drop('y', axis=1)  
y = df['y']
smote = SMOTE(random_state=42,)
F1={}
auc = {}
result = {}

for a in {"gini","entropy"}:
    for b in range (3,11,1):
        print(a,b)
        models = DecisionTreeClassifier(random_state=42,max_depth=b, criterion=a)
        for i in range(1,5,1):
            print(f"Test size: {i/10:.1f}, F1, AUC, Accuracy")
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=i/10, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            models.fit(X_resampled, y_resampled)
            y_pred = models.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            AUC= roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"DT,{f1:.3f},{AUC:.3f},{accuracy:.3f}")
            F1 [(a,b,i)] = f1
            auc [(a,b,i)] = AUC
            result [(a,b,i)] = accuracy
        print("")

for a in {"gini","entropy"}:
    for b in {None}:
        print(a,b)
        models = DecisionTreeClassifier(random_state=42,max_depth=b, criterion=a)
        for i in range(1,5,1):
            print(f"Test size: {i/10:.1f}, F1, AUC, Accuracy")
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=i/10, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            models.fit(X_resampled, y_resampled)
            y_pred = models.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            AUC= roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"DT,{f1:.3f},{AUC:.3f},{accuracy:.3f}")
            F1 [(a,b,i)] = f1
            auc [(a,b,i)] = AUC
            result [(a,b,i)] = accuracy
        print("")

sorted_f1 = sorted(F1.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_f1:
    print(f"F1: {value:.3f} for params {key}")