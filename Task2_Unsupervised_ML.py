from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import datasets

import pandas as ps
import matplotlib.pyplot as plt

# lets get the iris data
iris = datasets.load_iris()

iris_df = ps.DataFrame(iris.data, columns = iris.feature_names)
print("--------- Printing iris feature names -----------")
print(iris_df.head(10)) # See the first 5 rows

#to find the optimum cluster.

x = iris_df.iloc[:,[0,1,2,3]].values

print('--- x = -----------')
print(x)