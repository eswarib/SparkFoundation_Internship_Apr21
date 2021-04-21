from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib.pyplot as plt

# lets get the iris data
iris = datasets.load_iris()

iris_df = ps.DataFrame(iris.data, columns = iris.feature_names)
print("--------- Printing iris data -----------")
print(iris_df.head(10)) # See the first 5 rows

#to find the optimum cluster.

x = iris_df.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11) :
    #max_iter - how many iterations done for a single run of K-Means algorithm
    #n_init = how many times the algorithm is run with different centroid seeds
    #random_state - determines random number generation for centroids initialisation will be chosen randomly. this parameter defines the randomness
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    #lets find WCSS - Within Cluster Sum of Squares 
    wcss.append(kmeans.inertia_)
    average(wcss)

import matplotlib.pyplot as plt

#let's plot the WCSS to find the optimum cluster
# in elbow method, we assign each sample to a cluster,then find the WCSS
# for every value of n - number of clusters, WCSS value improves and then stabilizes
# we are interested in finding the point from which there is not major change in WCSS

fig_elbow, ax = plt.subplots(1,1,figsize=(15, 5))

x_axis = list(range(1,11))

ax.plot(x_axis,wcss)
ax.set_title('Elbow Method')
ax.set_xlabel("Number of clusters")
ax.set_ylabel('WCSS')
ax.set_xticks(x_axis)

ax.axvline(x=3,linestyle='--',c='red')

plt.show()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15, 10))
ax1.scatter(x[:, 0], x[:, 1],
            s = 30, c = 'red', label = 'Iris-setosa')



#let's calculate the predicted cluster each sample belongs to based on the model deter with no of clusters = 3
kmeans = KMeans(n_clusters = 3, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)

print('Cluster index predicted for each sample in the dataset')
print(y_kmeans)

#visualising the clusters on the fisrt two columns
ax2.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
            s = 30, c = 'red', label = 'Iris-setosa')

ax2.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
            s = 30, c = 'blue', label = 'Iris-versicolor')

ax2.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 30, c = 'green', label = 'Iris-viriginica')

ax2.legend()
centroids = kmeans.cluster_centers_
# lets mark the centroids

ax2.scatter(centroids[:,0],centroids[:,1], s = 150, c = 'purple')

