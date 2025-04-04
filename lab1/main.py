import csv

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import umap.umap_ as umap

mob_list = []

data = pd.read_csv("train.csv")
print(data.head())

print("Dataset shape:", data.shape)
print(data.isnull().any().any())


# Готовим данные для метода
x = data.loc[:, ['battery_power' ,'blue' ,'clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi','price_range']].values
print(x.shape)
print(x)


# Стандартизация данных
scaler = StandardScaler()
scaler.fit(data)
scaled_features = scaler.transform(data)
scaled_data = pd.DataFrame(scaled_features, columns = data.columns)
print("Стандартизированные данные:", scaled_features)
print(scaled_data)

#------------------------------KMEANS--------------------------------

#Метод логтя
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
clusters = kl.elbow

print("Число кластеров", clusters)

#Кластеризация k-средних
kmeans = KMeans(
    init="random",
    n_clusters=clusters,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(scaled_features)
labels_kmeans = kmeans.labels_
np.savetxt('kmeans-labels.csv', labels_kmeans, delimiter=',')

print(labels_kmeans)

print(data.columns)

processed = []
indices = {
    'battery_power': 0,
    'blue': 1,
    'clock_speed': 2,
    'dual_sim': 3,
    'fc': 4,
    'four_g': 5,
    'int_memory': 6,
    'm_dep': 7,
    'mobile_wt': 8,
    'n_cores': 9,
    'pc': 10,
    'px_height': 11,
    'px_width': 12,
    'ram': 13,
    'sc_h': 14,
    'sc_w': 15,
    'talk_time': 16,
    'three_g': 17,
    'touch_screen': 18,
    'wifi': 19,
    'price_range': 20
}
counter = 1
for i in data.columns:
    processed.append(i)
    for j in data.columns:
        if j not in processed:
            fig = plt.figure()
            plt.scatter(x[:, indices[i]], x[:, indices[j]], c=labels_kmeans * 10, cmap="plasma")
            plt.xlabel(i)
            plt.ylabel(j)
            #plt.show()

            plt.close(fig)
            counter += 1

#------------------------------DBSCAN--------------------------------

#Поиск epsion для метода DBSCAN
neighb = NearestNeighbors(n_neighbors=2)
nbrs=neighb.fit(scaled_data)
distances,indices=nbrs.kneighbors(scaled_data)

distances = np.sort(distances, axis = 0)
distances = distances[:, 1]
plt.rcParams['figure.figsize'] = (5,3)
plt.plot(distances)
plt.show()
eps = 2.6
MinPoints = 2

#метод DBSCAN
dbscan = DBSCAN(eps = eps,
                min_samples = MinPoints,
                algorithm='ball_tree',
                metric='minkowski',
                leaf_size=90,
                p=2).fit(scaled_data)


labels_dbscan = dbscan.labels_
np.savetxt('dbscan-labels.csv', labels_dbscan, delimiter=',')

counter = 1
for i in data.columns:
    processed.append(i)
    for j in data.columns:
        if j not in processed:
            fig = plt.figure()
            plt.scatter(x[:, indices[i]], x[:, indices[j]], c=labels_dbscan * 10, cmap="plasma")
            plt.xlabel(i)
            plt.ylabel(j)
            #plt.show()

            plt.close(fig)
            counter += 1

print(labels_kmeans)
print(labels_dbscan)


reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_data)

fig = plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c = labels_kmeans, cmap= "plasma")
plt.xlabel('UMAP reduced 1')
plt.ylabel('UMAP reduced 2')
plt.savefig('UMAP:kmeans')
plt.close(fig)

fig = plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c = labels_dbscan, cmap= "plasma")
plt.xlabel('UMAP reduced 1')
plt.ylabel('UMAP reduced 2')
plt.savefig('UMAP:dbscan')
plt.close(fig)


pca = PCA(n_components = 2)
XPCAreduced = pca.fit_transform(scaled_data)

fig = plt.figure()
plt.scatter(XPCAreduced[:, 0], XPCAreduced[:, 1], c = labels_kmeans, cmap= "plasma")
plt.xlabel('PCA reduced 1')
plt.ylabel('PCA reduced 2')
plt.savefig('PCA:kmeans')
plt.close(fig)

fig = plt.figure()
plt.scatter(XPCAreduced[:, 0], XPCAreduced[:, 1], c = labels_dbscan, cmap= "plasma")
plt.xlabel('PCA reduced 1')
plt.ylabel('PCA reduced 2')
plt.savefig('PCA:dbscan')
plt.close(fig)




'''
def __kmeans___(data, scaled_data):
    #Метод логтя
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    clusters = kl.elbow

    print(clusters)

    #Кластеризация k-средних
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(scaled_features)
    labels = kmeans.labels_

    print(labels)

    print(data.columns)

    processed =[]
    indices = {
        'battery_power': 0,
        'blue': 1,
        'clock_speed': 2,
        'dual_sim': 3,
        'fc': 4,
        'four_g': 5,
        'int_memory': 6,
        'm_dep': 7,
        'mobile_wt': 8,
        'n_cores': 9,
        'pc': 10,
        'px_height': 11,
        'px_width': 12,
        'ram': 13,
        'sc_h': 14,
        'sc_w': 15,
        'talk_time': 16,
        'three_g': 17,
        'touch_screen': 18,
        'wifi': 19,
        'price_range': 20
    }
    counter = 1
    for i in data.columns:
        processed.append(i)
        for j in data.columns:
            if j not in processed:
                fig = plt.figure()
                plt.scatter(x[:, indices[i]], x[:,indices[j]], c = labels*10, cmap= "plasma")
                plt.xlabel(i)
                plt.ylabel(j)
                plt.show()
                #plt.savefig('out-kmeans/' + str(counter) + ') ' + i + '-' + j)
                plt.close(fig)
                counter += 1
'''

'''
def __DBSCAN__(data, scaled_data):
    neighb = NearestNeighbors(n_neighbors=2)
    nbrs=neighb.fit(scaled_data)
    distances,indices=nbrs.kneighbors(scaled_data)

    distances = np.sort(distances, axis = 0)
    distances = distances[:, 1]
    plt.rcParams['figure.figsize'] = (5,3)
    plt.plot(distances)
    plt.show()
    #print(neighb)
    #print(nbrs)
    eps = 3.3
    MinPoints = 5

    dbscan = DBSCAN(eps = eps,
                    min_samples = MinPoints,
                    algorithm='ball_tree',
                    metric='minkowski',
                    leaf_size=90,
                    p=2).fit(scaled_data)


    labels = dbscan.labels_

    processed = []
    indices = {
        'battery_power': 0,
        'blue': 1,
        'clock_speed': 2,
        'dual_sim': 3,
        'fc': 4,
        'four_g': 5,
        'int_memory': 6,
        'm_dep': 7,
        'mobile_wt': 8,
        'n_cores': 9,
        'pc': 10,
        'px_height': 11,
        'px_width': 12,
        'ram': 13,
        'sc_h': 14,
        'sc_w': 15,
        'talk_time': 16,
        'three_g': 17,
        'touch_screen': 18,
        'wifi': 19,
        'price_range': 20
    }
    counter = 1
    for i in data.columns:
        processed.append(i)
        for j in data.columns:
            if j not in processed:
                fig = plt.figure()
                plt.scatter(x[:, indices[i]], x[:, indices[j]], c=labels * 10, cmap="plasma")
                plt.xlabel(i)
                plt.ylabel(j)
                plt.show()
                # plt.savefig('out-kmeans/' + str(counter) + ') ' + i + '-' + j)
                plt.close(fig)
                counter += 1
'''



#__kmeans___(data, scaled_data)
#__DBSCAN__(data, scaled_data)