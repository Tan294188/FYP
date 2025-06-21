import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Prediction Term Deposit Subscription", layout="wide")


# Load dataset
data = pd.read_csv('banking-marketing_combined.csv')

# Load dataset
raw_df = pd.read_csv('banking-marketing_combined.csv')

def preprocess_data(df):
    df = df[~df['poutcome'].isin(['other'])]
    df = df[~df['job'].isin(['unknown'])]
    df = df[~df['education'].isin(['unknown'])]
    df = df[df['pdays'] != -1]
    df = df[df['balance'] >= 0]

    kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    df['age'] = kbin.fit_transform(df[['age']])
    kbin2 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
    df['campaign'] = kbin2.fit_transform(df[['campaign']])

    df['level'] = pd.cut(
        df['balance'],
        bins=[df['balance'].min(), df['balance'].quantile(0.2), df['balance'].quantile(0.7), float('inf')],
        labels=['low', 'middle', 'high']
    )
    df['level'] = df['level'].fillna('low')
    df['burden'] = np.where((df['housing'] == 'yes') | (df['loan'] == 'yes'), 1, 0)

    df.drop(['level', 'burden'], axis=1, inplace=True)

    df['income'] = np.select([
        df['job'].isin(['entrepreneur', 'self-employed', 'management', 'admin.']),
        df['job'].isin(['technician', 'services']),
        df['job'].isin(['blue-collar', 'housemaid'])
    ], ['high', 'moderate', 'low'], default='none')

    df['risk'] = np.select([
        df['job'].isin(['entrepreneur', 'self-employed']),
        df['job'].isin(['management', 'admin.', 'technician', 'services']),
        df['job'].isin(['blue-collar', 'housemaid'])
    ], ['high', 'moderate', 'low'], default='dependent')

    df.drop(['income', 'risk'], axis=1, inplace=True)

    df['poutcome'] = df['poutcome'].replace('unknown', 'failure')
    df.drop(['day', 'month', 'contact'], axis=1, inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    return df

@st.cache_resource

def train_model():
    df = preprocess_data(raw_df.copy())
    X = df.drop('y', axis=1)
    y = df['y']
    smote = SMOTE(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    model =  RandomForestClassifier(n_estimators=85, random_state=42, max_depth=None, bootstrap=False)
    model.fit(X_resampled, y_resampled)
    return model

model = train_model()
# Streamlit app

# Navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Descriptive Analysis", "Predictive Analysis", "User Prediction", "Prescriptive Analysis"])

if page == "Home":
    st.title("Prediction Term Deposit Subscription")

    st.markdown("""
    [Link to Dataset on Hugging Face](https://huggingface.co/datasets/Andyrasika/banking-marketing?sql=--+The+SQL+console+is+powered+by+DuckDB+WASM+and+runs+entirely+in+the+browser.%0A--+Get+started+by+typing+a+query+or+selecting+a+view+from+the+options+below.%0ASELECT+*+FROM+test+LIMIT+10%3B)

    **About Dataset**

    Term deposits are a major source of income for a bank. A term deposit is a cash investment held at a financial institution. Term deposit investments typically have short maturities ranging from one month to a few years and demand various minimum deposits. Varied time periods for term deposits result in varied return rates. In general, the interest rate offered increases with the length of the contract. To increase the subscription of term deposit, telephonic marketing campaigns be involved. Telephonic marketing campaigns still remain one of the most effective way to reach out to people. 

    The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y). This dataset consists of 49,732 rows and 18 columns ordered by date (from May 2008 to November 2010).
    """)

    st.subheader("Dataset Variables")
    var_table = pd.DataFrame({
        "Variable": ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"],
        "Data Type": ["Numerical", "Categorical", "Categorical", "Categorical", "Binary", "Numerical", "Binary", "Binary", "Categorical", "Numerical", "Categorical", "Numerical", "Numerical", "Numerical", "Numerical", "Categorical", "Binary"],
        "Explanation": [
            "Age of client", "Type of job", "Marital Status", "Education level of client", "Has credit in default?",
            "Average yearly balance, in euros", "Has housing loan?", "Has personal loan?", "Contact communication type",
            "Last contact day of the month", "Last contact month of the year", "Last contact duration, in seconds",
            "Number of contacts performed during this campaign and for this client",
            "Number of days that passed by after the client was last contacted from a previous campaign",
            "Number of contacts performed before this campaign and for this client",
            "Outcome of the previous marketing campaign", "Has the client subscribed a term deposit?"
        ]
    })
    st.dataframe(var_table)

    st.subheader("Dataset Sample")
    st.data_editor(data.head(20), use_container_width=True, disabled=True, num_rows="dynamic")

elif page == "Descriptive Analysis":
    st.title("Descriptive Analysis")

    st.subheader("Pie Chart of Target Variable 'y'")
    pie_data = data['y'].value_counts(normalize=True)
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig1)

    st.subheader("Categorical Variables with Target 'y'")
    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    for var in cat_vars:
        st.write(f"Variable: {var}")
        st.dataframe(pd.crosstab(data[var], data['y'], normalize='index'))

    st.subheader("Frequency Bar Chart for 'month'")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=data, x='month', ax=ax2, order=data['month'].value_counts().index)
    st.pyplot(fig2)

    st.subheader("Histogram of Numerical Data")
    num_vars = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for var in num_vars:
        fig, ax = plt.subplots()
        sns.histplot(data[var], kde=True, ax=ax)
        mean_val = data[var].mean()
        std_val = data[var].std()
        ax.axvline(mean_val, color='r', linestyle='--', label='Mean')
        ax.axvline(mean_val + std_val, color='g', linestyle='--', label='Mean + 1 SD')
        ax.axvline(mean_val - std_val, color='g', linestyle='--', label='Mean - 1 SD')
        ax.legend()
        st.pyplot(fig)

elif page == "Predictive Analysis":
    st.title("Predictive Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initial Performance")
        st.markdown("#### Test size: 0.1")
        df1 = pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.577, 0.676, 0.621, 0.685, 0.768],
            'AUC': [0.723, 0.8, 0.76, 0.803, 0.862],
            'Accuracy': [0.757, 0.811, 0.773, 0.822, 0.871]
        })
        st.dataframe(df1, use_container_width=True)

        st.markdown("#### Test size: 0.2")
        df2 = pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.531, 0.634, 0.575, 0.638, 0.699],
            'AUC': [0.694, 0.774, 0.732, 0.775, 0.814],
            'Accuracy': [0.728, 0.79, 0.742, 0.795, 0.839]
        })
        st.dataframe(df2, use_container_width=True)

        st.markdown("#### Test size: 0.3")
        df3 = pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.519, 0.625, 0.556, 0.633, 0.696],
            'AUC': [0.692, 0.773, 0.722, 0.775, 0.818],
            'Accuracy': [0.718, 0.787, 0.74, 0.8, 0.838]
        })
        st.dataframe(df3, use_container_width=True)

        st.markdown("#### Test size: 0.4")
        df4 = pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.512, 0.635, 0.567, 0.636, 0.701],
            'AUC': [0.679, 0.776, 0.724, 0.771, 0.818],
            'Accuracy': [0.709, 0.789, 0.742, 0.798, 0.835]
        })
        st.dataframe(df4, use_container_width=True)

    with col2:
        st.subheader("Improved Performance")
        st.markdown("#### Test size: 0.1")
        st.dataframe(pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.71, 0.712, 0.697, 0.715, 0.774],
            'AUC': [0.834, 0.828, 0.817, 0.826, 0.859],
            'Accuracy': [0.822, 0.831, 0.822, 0.838, 0.879]
        }), use_container_width=True)

        st.markdown("#### Test size: 0.2")
        st.dataframe(pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.659, 0.675, 0.65, 0.68, 0.717],
            'AUC': [0.798, 0.805, 0.787, 0.807, 0.822],
            'Accuracy': [0.798, 0.814, 0.798, 0.82, 0.854]
        }), use_container_width=True)

        st.markdown("#### Test size: 0.3")
        st.dataframe(pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.625, 0.659, 0.65, 0.644, 0.69],
            'AUC': [0.779, 0.799, 0.787, 0.779, 0.809],
            'Accuracy': [0.777, 0.808, 0.798, 0.811, 0.84]
        }), use_container_width=True)

        st.markdown("#### Test size: 0.4")
        st.dataframe(pd.DataFrame({
            'Model': ['KNN', 'Linear Regression', 'SVM', 'Decision Tree', 'Random Forest'],
            'F1': [0.642, 0.662, 0.642, 0.683, 0.713],
            'AUC': [0.787, 0.794, 0.781, 0.812, 0.819],
            'Accuracy': [0.781, 0.807, 0.79, 0.818, 0.85]
        }), use_container_width=True)

elif page == "Prescriptive Analysis":
    st.title("Prescriptive Analysis")

    st.markdown("""
    ### Model Selection
    Random Forest performs best, with 80:20 data partition recommended over 90:10 due to dataset size and stability concerns. While 60:40 offers more training data, it limits learning minority class features.

    ### Data-Driven Decision Making
    To improve term deposit subscriptions:
    - Avoid marketing to clients with negative balances, as they are unlikely to subscribe.
    - Increase return rates to enhance bank competitiveness and attract more clients.

    ### Future Improvement
    Model performance is below 80% due to:
    - Poor data quality—imbalanced data and missing key variables like gender and year.
    - Limited preprocessing techniques—resource constraints prevent testing all combinations.

    Improvements include expanding dataset variables and testing additional preprocessing techniques.

    ### Risk Assessment
    - **Misclassification Risk**: Lower model accuracy may lead to incorrect client targeting; validation techniques can mitigate this.
    - **Misunderstanding Risk**: New features may misalign with client data, causing biased predictions; careful feature selection is crucial.

    Refinements in data and feature selection will enhance model reliability for banking applications.
    """)
    
elif page == "User Prediction":
    st.title("Predict Term Deposit Subscription")
    df = preprocess_data(raw_df.copy())
    feature_columns = df.drop('y', axis=1).columns

    st.subheader("Input Client Data")
    user_input = {}

    for col in feature_columns:
        unique_vals = sorted(df[col].unique())
        if len(unique_vals) <= 10:
            user_input[col] = st.selectbox(col, unique_vals)
        else:
            user_input[col] = st.slider(col, int(min(unique_vals)), int(max(unique_vals)), int(min(unique_vals)))

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
    
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {'Subscribed' if prediction == 1 else 'Not Subscribed'} (Probability: {proba:.2f})")
# streamlit run website.py
