import pandas as pd
df=pd.read_csv("data\\raw\\churn.csv")
print(df.isnull().sum())
print(df.describe())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print(df['TechSupport'].unique())
df=df.drop(columns=['customerID'])

#for eda
df_raw=df.copy()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#label encoding for binary categorical columns
df['Churn']=le.fit_transform(df['Churn'])
df['gender']=le.fit_transform(df['gender'])
# df['SeniorCitizen']=le.fit_transform(df['SeniorCitizen'])
df['Partner']=le.fit_transform(df['Partner'])
df['Dependents']=le.fit_transform(df['Dependents'])
df['PhoneService']=le.fit_transform(df['PhoneService'])
df['PaperlessBilling']=le.fit_transform(df['PaperlessBilling'])

#one hot encoding for categorical columns with more than 2 categories
df=pd.get_dummies(df,columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'])

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

print(df.dtypes)
print("Number of object columns:",len(df.select_dtypes(include=['object']).columns))

# save cleaned data for downstream scripts
df.to_csv("data\\processed\\churn_clean.csv", index=False)
print('Wrote churn_clean.csv with', len(df), 'rows')

df_raw.to_csv("data\\processed\\churn_eda.csv", index=False)
print('Wrote churn_eda.csv with', len(df_raw), 'rows')