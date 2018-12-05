#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import random
#Data
products = range(10000)
users = range(1000)
purchases = []
for p in range(100000):
    u = random.choice(users)
    p = random.choice(products)
    purchases.append((u,p))

data = pd.DataFrame(purchases).values

#Apply the Kmeans-model 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,9):
    kmeans = KMeans(n_clusters = i, init='k-means++',random_state = 0,)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,9),wcss, marker = 'o')
plt.title('THE ELBOW METHOD')
plt.xlabel('NO OF CLUSTERS')
plt.ylabel('PRODUCTS')
plt.show()
kmeans = KMeans(n_clusters=3, init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(data)
#visualising the data
plt.scatter(data[y_kmeans == 0,0], data[y_kmeans == 0,1], s = 100, c = 'red')
plt.scatter(data[y_kmeans == 1,0], data[y_kmeans == 1,1], s = 100, c = 'blue')
plt.scatter(data[y_kmeans == 2,0], data[y_kmeans == 2,1], s = 100, c = 'black')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow')
plt.title('Purchases')
plt.xlabel('Users')
plt.ylabel('Products')
plt.show()












