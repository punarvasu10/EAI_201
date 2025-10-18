import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== EXPLORATORY DATA ANALYSIS ===")
print("\n1. Dataset Shape:", df.shape)
print("\n2. First 5 rows:")
print(df.head())

print("\n3. Data Types & Missing Values:")
print(df.info())

print("\n4. Missing Values Summary:")
print(df.isnull().sum())

print("\n5. Basic Statistics:")
print(df.describe())

# Visualizations
plt.figure(figsize=(15, 10))

# Distribution of Age
plt.subplot(2, 3, 1)
df['Age'].hist(bins=20)
plt.title('Age Distribution')

# Survival by Sex
plt.subplot(2, 3, 2)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')

# Survival by Pclass
plt.subplot(2, 3, 3)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Pclass')

# Fare distribution
plt.subplot(2, 3, 4)
df['Fare'].hist(bins=20)
plt.title('Fare Distribution')

# Survival by Embarked
plt.subplot(2, 3, 5)
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival by Embarked')

plt.tight_layout()
plt.show()

print("\n=== DATA CLEANING ===")
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop irrelevant columns
df_clean = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)  # Keep Name for title extraction

print("Missing values after cleaning:")
print(df_clean.isnull().sum())

print("\n=== FEATURE ENGINEERING ===")
# Family Size
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1

# Extract Title from Name - FIXED ESCAPE SEQUENCE
df_clean['Title'] = df_clean['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_clean['Title'] = df_clean['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_clean['Title'] = df_clean['Title'].replace('Mlle', 'Miss')
df_clean['Title'] = df_clean['Title'].replace('Ms', 'Miss')
df_clean['Title'] = df_clean['Title'].replace('Mme', 'Mrs')

print("Title counts:")
print(df_clean['Title'].value_counts())

# Convert categorical features
label_encoder = LabelEncoder()
df_clean['Sex_encoded'] = label_encoder.fit_transform(df_clean['Sex'])
df_clean['Embarked_encoded'] = label_encoder.fit_transform(df_clean['Embarked'])
df_clean['Title_encoded'] = label_encoder.fit_transform(df_clean['Title'])

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df_clean, columns=['Sex', 'Embarked', 'Title', 'Pclass'], drop_first=True)

# Drop original Name column after title extraction
df_encoded = df_encoded.drop('Name', axis=1)

print("\n=== FINAL DATASET ===")
print("Shape:", df_encoded.shape)
print("\nColumns:", df_encoded.columns.tolist())
print("\nFirst 3 rows:")
print(df_encoded.head(3))

print("\n=== DATA SPLITTING ===")
# Prepare features and target
X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print("\nData ready for modeling!")

# Show survival rates by key features
print("\n=== SURVIVAL ANALYSIS ===")
print("\nSurvival Rate by Sex:")
print(df.groupby('Sex')['Survived'].mean())

print("\nSurvival Rate by Pclass:")
print(df.groupby('Pclass')['Survived'].mean())

print("\nSurvival Rate by Family Size:")
print(df_clean.groupby('FamilySize')['Survived'].mean())