import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from CustomScaler import CustomScaler
from sklearn import metrics

# Data Pre-processing
# *************************************************************************************************************
# Load data
# *****************************************************************************************
data = pd.read_csv('Absenteeism_data.csv')
# print(data)

df = data.copy()
# print(df.info())

df = df.drop(['ID'], axis=1)
# print(df)

# Create dummy variables for Reasons
# *****************************************************************************************
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
# reason_columns['check'] = reason_columns.sum(axis=1)
# print(reason_columns['check'].sum(axis=0))

# Group the Reasons for Absence
# *****************************************************************************************
df = df.drop(['Reason for Absence'], axis=1)

reason_type1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type4 = reason_columns.loc[:, 22:].max(axis=1)

df = pd.concat([df, reason_type1, reason_type2, reason_type3, reason_type4], axis=1)

# Rename the columns
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
                'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2',
                'Reason_3', 'Reason_4']
df.columns = column_names

# Reorder the columns
# *****************************************************************************************
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                          'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']
df = df[column_names_reordered]

# Reformat the 'Date'
# *****************************************************************************************
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# print(df['Date'][0])

# Extract month value
# *****************************************************************************************
list_month = []
for i in range(700):
    list_month.append(df['Date'][i].month)

df['Month Value'] = list_month


# print(df)

# Extract the Day of the Week
# *****************************************************************************************
def date_to_weekday(date_value):
    return date_value.weekday()


df['Day of the Week'] = df['Date'].apply(date_to_weekday)
df.drop(['Date'], axis=1)
# print(df.columns)
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                          'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
                          'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
df = df[column_names_reordered]

# Create dummy variables for Educations (0 or 1)
# *****************************************************************************************
df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

# Logistic Regression
# *************************************************************************************************************
# Create targets
# *****************************************************************************************
targets = np.where(df['Absenteeism Time in Hours'] > df['Absenteeism Time in Hours'].median(), 1, 0)
df['Excessive Absenteeism'] = targets

data_with_targets = df.drop(['Absenteeism Time in Hours'], axis=1)

# Select inputs
# *****************************************************************************************
unscaled_inputs = data_with_targets.iloc[:, :-1]
# print(unscaled_inputs.columns)

# Standardize the data
# *****************************************************************************************
absenteeism_scaler = StandardScaler()
# columns_to_scale = ['Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
#                     'Daily Work Load Average', 'Body Mass Index' 'Education' 'Children' 'Pets']
# absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
# print(scaled_inputs)

# Split data into training and testing
# *****************************************************************************************
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
# print(len(train_test_split(scaled_inputs, targets)[2]))

# Training the model
# *****************************************************************************************
reg = LogisticRegression()
reg.fit(x_train, y_train)
# print(reg.score(x_train, y_train))

# Manually check the accuracy
# *****************************************************************************************
# model_outputs = reg.predict(x_train)
# print(np.sum(model_outputs == y_train) / model_outputs.shape[0])

# Finding the intercept and coefficient
feature_names = unscaled_inputs.columns.values
# print(feature_names)
# print(reg.coef_[0])
summary_tables = pd.DataFrame(columns=['Feature Names'], data=feature_names)
summary_tables['Coefficient'] = reg.coef_[0]
summary_tables['Odds_ratio'] = np.exp(summary_tables.Coefficient)

# print(summary_tables.sort_values(by=['Odds_ratio'], ascending=False))

# Tesing the model
# *****************************************************************************************
print(reg.score(x_test, y_test))
predicted_proba = reg.predict_proba(x_test)
print(predicted_proba)


