import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

df = pd.read_csv('airbnb.csv')
print(df.isnull().sum()) # print the number of missing values in each column

missing_data = df.isnull().mean() * 100
print(missing_data) # print the percentage of missing values in each column

print(missing_data[missing_data > 0]) # print the columns with missing values
plt.figure(figsize=(6, 4))
plt.bar(missing_data[missing_data > 0].index, missing_data[missing_data > 0], color='blue')
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Percentage (%)')
plt.xlabel('Columns that have missing values')
plt.show()

correlationMatrix = df.select_dtypes(include=["int","float"]).corr()
print(correlationMatrix) # print the correlation matrix

seaborn.heatmap(correlationMatrix,annot=True)
plt.show() # Plot the correlation matrix

df.fillna({'name': df['name'].mode()[0]}, inplace=True) 
df.fillna({'host_name': df['host_name'].mode()[0]}, inplace=True)
df.fillna({'last_review': df['last_review'].mode()[0]}, inplace=True)
df.fillna({'reviews_per_month': df['reviews_per_month'].median()}, inplace=True)

# drop the columns that are not needed for clustering
df = df.drop (columns=['id', 'name', 'host_id', 'host_name', 'last_review'])

encoder = OneHotEncoder(sparse_output=False)
newArray = encoder.fit_transform(df[["neighbourhood_group", "neighbourhood", "room_type"]])
cols = encoder.get_feature_names_out()
newDf = pd.DataFrame(newArray,columns=cols,index=df.index)
newDataFrame = pd.concat([df.drop(columns=["neighbourhood_group", "neighbourhood", "room_type"],axis=1),newDf],axis=1)
print(newDataFrame.head())

for col in newDataFrame.columns:
    maen = newDataFrame[col].mean()
    std = newDataFrame[col].std()
    newDataFrame[col] = (newDataFrame[col] - maen) / std
print(newDataFrame.head())
pca = PCA(n_components=0.95)
ReductionDf = pca.fit_transform(newDataFrame)
print(ReductionDf.shape)
X_train, remainingData = train_test_split(ReductionDf, test_size=0.4, random_state=42)
X_val, X_test = train_test_split(remainingData, test_size=0.5, random_state=42)
# print the shapes of the train, validation, and test sets to check wheather the columns are equal in every set
print("Train Data shape:", X_train.shape)
print("Validation Data shape:", X_val.shape)
print("Test Data shape:", X_test.shape)

class KMeans:
    def __init__(self, k, n_clusters=3, max_iter=100, tol=1e-4):
        self.k = k
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    # Choose # of clusters desired, k. ==> Randomly select k points from the dataset as initial centroids
    def initializeCentroids(self, X):
        randomPoints = np.random.choice(X.shape[0], size=self.k, replace=False)# Used replace=False to make sure that centroids are unique
        self.centroids = X[randomPoints]
    
    # Assign each point to its closest centroids
    def assignPointsToCentroids(self, X):
        nearestCentroid = []
        for i in X:
            distances = []
            for j in self.centroids:
                euclideanDistance = np.linalg.norm(i - j)
                distances.append(euclideanDistance)
            nearestCentroidIndex = np.argmin(distances)
            nearestCentroid.append(nearestCentroidIndex)
        self.labels = np.array(nearestCentroid)
    
    # Re-compute centroids
    def recomputeCentroids(self, X):
        for i in range(self.k):
            dataPoints = X[self.labels == i]
            if len(dataPoints) > 0:
                self.centroids[i] = dataPoints.mean(axis=0)
            else:
                self.centroids[i] = self.centroids[i]

    # check convergence
    def checkSignificantChange(self, oldCentroids):
        difference = np.abs(self.centroids - oldCentroids)
        boolean = np.all(difference < self.tol)
        return boolean
    
    # Fit the model to the data
    def fit(self, X):
        self.initializeCentroids(X)
        for i in range(self.max_iter):
            self.assignPointsToCentroids(X)
            oldCentroids = self.centroids.copy()
            self.recomputeCentroids(X)
            if self.checkSignificantChange(oldCentroids):
                print(f"Converged after {i+1} iterations.")
                break

    # Predict the cluster labels for new data points
    def predict(self, X):
        self.assignPointsToCentroids(X)
        return self.labels
    
    # getters to get the centroids and labels
    def getCentroids(self):
        return self.centroids
    
    def getLabels(self):
        return self.labels
    
Kmeans = KMeans(k=3)
Kmeans.fit(X_train)
print("Centroids:")
print(Kmeans.getCentroids()) # print the centroids of the clusters
print("Labels:")
print(Kmeans.getLabels()) # print the labels of the clusters

from sklearn.cluster import KMeans as SKMeans
mse = []
# test for k = 1 to 10
for i in range(1,11):
    skmeans = SKMeans(n_clusters=i, random_state=42)
    skmeans.fit(X_train)
    mse.append(skmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1,11), mse, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('MSE (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.grid()
plt.show()

bestK = 4
finalKMeanns = KMeans(bestK)
finalKMeanns.fit(X_train)
labels = finalKMeanns.predict(X_train)
pca = PCA(n_components=2)
reducedToTwoDimension = pca.fit_transform(X_train)
scatter = plt.scatter(reducedToTwoDimension[:, 0], reducedToTwoDimension[:, 1], c=labels, edgecolors='k')
plt.title('Cluster Visualization (k=4) in 2D')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

class bisectingKMeans:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.labels = None

    def sse(self, cluster):
        centroid = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - centroid, axis=1)
        totalSSE = np.sum(distances**2)
        return totalSSE
    
    def fit(self, X):
        clusters = [X]
        while len(clusters) < self.k:
            sseList = [self.sse(cluster) for cluster in clusters]
            highestSSEIndex = np.argmax(sseList)
            highestSSECluster = clusters.pop(highestSSEIndex)
            bkmeans = KMeans(k=2, max_iter=self.max_iter, tol=self.tol)
            bkmeans.fit(highestSSECluster)
            newClusters = [highestSSECluster[bkmeans.getLabels() == 0], highestSSECluster[bkmeans.getLabels() == 1]]
            clusters.extend(newClusters)
            self.centroids.extend(bkmeans.centroids)
        labels = []
        for cluster in clusters:
            labels.append(bkmeans.predict(cluster))
        self.labels = np.concatenate(labels)

    def predict(self, X):
        labels = []
        for i in X:
            distances = []
            for j in self.centroids:
                euclideanDistance = np.linalg.norm(i - j)
                distances.append(euclideanDistance)
            labels.append(np.argmin(distances))
        self.labels = np.array(labels)

bisectingkmeans = bisectingKMeans(k=4)
bisectingkmeans.fit(X_train)
bklabels = bisectingkmeans.predict(X_train)
pca = PCA(n_components=2)
reducedToTwoDimensionBK = pca.fit_transform(X_train)
scatter = plt.scatter(reducedToTwoDimensionBK[:, 0], reducedToTwoDimensionBK[:, 1], c=bklabels, cmap='plasma', s=30, edgecolors='k')
plt.title('Bisecting K-Means Cluster Visualization (k=4) in 2D')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.show()