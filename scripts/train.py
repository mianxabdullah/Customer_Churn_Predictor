import pandas as pd
df = pd.read_csv("data\\processed\\churn_clean.csv")

from sklearn.model_selection import train_test_split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])
