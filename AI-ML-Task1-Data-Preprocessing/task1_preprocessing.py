import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load Dataset
df = pd.read_csv("titanic.csv")
print("✅ Dataset Loaded Successfully!")
print(df.head())
print(df.info())

# Step 2: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.dropna(thresh=5, inplace=True)

# Step 3: Encode Categorical Variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 4: Standardize Numerical Features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 5: Detect & Remove Outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= (Q1 - 1.5 * IQR)) & (df['Fare'] <= (Q3 + 1.5 * IQR))]

# Step 6: Visualize After Cleaning
sns.boxplot(x=df['Fare'])
plt.title("Boxplot After Removing Outliers")
plt.show()

# Step 7: Save Cleaned Dataset
df.to_csv("titanic_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as titanic_cleaned.csv")
