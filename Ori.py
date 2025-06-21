import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df= pd.read_csv('banking-marketing_combined.csv')

df = df[~df['poutcome'].isin(['other'])]
df =df[~df['job'].isin(['unknown'])]
df = df[~df['education'].isin(['unknown'])]
df = df[~(df['pdays'] == -1) ]
df = df[~(df['balance'] < 0)]

df['poutcome'] = df['poutcome'].replace('unknown', 'failure')

df.drop(['day', 'month','contact'], axis=1, inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Linear Regression': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', penalty='l2'),
    'SVM': SVC(kernel='rbf', C=10),
    'Decision Tree': DecisionTreeClassifier(random_state=42,max_depth=10), 
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
}
F1= {}
auc={}
results = {}

X = df.drop('y', axis=1)  
y = df['y']
smote = SMOTE(random_state=42)


for i in range(1,5,1):
    print(f"Test size: {i/10:.1f}, F1, AUC, Accuracy")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=i/10, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        F1[name] = f1
        AUC= roc_auc_score(y_test, y_pred)
        auc[name] = AUC
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name},{f1:.3f},{AUC:.3f},{accuracy:.3f}")
    print("")
    