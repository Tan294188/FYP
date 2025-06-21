import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('banking-marketing_combined.csv')

# Get the numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Get the categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Print the statistical summary of numerical data
print(df[numerical_cols].describe())

# Visualize the numerical data
for col in numerical_cols:
    df[col].plot(kind='hist')
    plt.title(col)
    plt.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
    plt.axvline(df[col].std(), color='green', linestyle='--', label=f'Std: {df[col].std():.2f}')
    plt.legend()
    plt.show()

# Plot a bar chart for the 'month' column
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df['month'].value_counts().loc[months].plot(kind='bar')
plt.title('Month Distribution')
plt.xlabel('Month')
plt.ylabel('Count')

for i, v in enumerate(df['month'].value_counts().loc[months]):
    plt.text(i, v + 3, str(v), color='black', ha='center')
plt.show()

# Print the statistical summary of categorical data
for col in categorical_cols:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print(pd.crosstab(df[col], df['y']))
    print()

print(pd.crosstab(df['default'], df['y']))
print(pd.crosstab(df['housing'], df['y']))
print(pd.crosstab(df['loan'], df['y']))
# Plot a pie chart for the 'y' column
df['y'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('y')
plt.show()