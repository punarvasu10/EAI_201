import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data directly from URL
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)
print("Data shape:", data.shape)
print(data.head())

# Preprocessing here we Droping ID column 
data.drop('customerID', axis=1, inplace=True)

# Handling the missing or blank TotalCharges.
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encode categorical columns
for col in data.columns:
    if data[col].dtypes == 'object':
        if len(data[col].unique()) == 2:
            data[col] = LabelEncoder().fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], drop_first=True)

# Standardize numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA - Reduce to 2 components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", round(sum(pca.explained_variance_ratio_)*100, 2), "%")

# Plot PCA scatter
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA - 2D Scatter Plot")
plt.show()

# K-Means - Find best number of clusters
inertia = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(pca_data)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2,11), inertia, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Silhouette Score check
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    preds = km.fit_predict(pca_data)
    print(f"k={k}, Silhouette Score={silhouette_score(pca_data, preds):.3f}")

# Apply K-Means with chosen k (e.g. 3)
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(pca_data)

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=data['Cluster'], palette='Set2')
plt.title("K-Means Clusters on PCA Data")
plt.show()

# Analyzing clusters in original data
summary = data.groupby('Cluster')[['MonthlyCharges','TotalCharges','tenure','Churn']].mean()
print("\nCluster Summary:")
print(summary)

# Visualize MonthlyCharges by cluster
plt.figure(figsize=(6,4))
sns.boxplot(x='Cluster', y='MonthlyCharges', data=data)
plt.title("Monthly Charges by Cluster")
plt.show()

# Visualize tenure by cluster
plt.figure(figsize=(6,4))
sns.boxplot(x='Cluster', y='tenure', data=data)
plt.title("Tenure by Cluster")
plt.show()