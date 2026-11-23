import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

iris_folder = Path(__file__).parent / 'images'
iris_folder.mkdir(exist_ok= True)

iris_df = sns.load_dataset('iris')
print(iris_df.head())
"""
Explore data
To cluster iris of multiple different species, 
we determine how many different types of iris species are in the dataset.
"""
print(iris_df['species'].unique())

"""
Check de count of each species type
"""
print(iris_df['species'].value_counts())

"""
We detected null values ​​in the dataset.
"""
print(iris_df.isna().sum())

"""
Since we have no null values, 
we can work with the same dataset without making any modifications.
We will copy all the data from the dataset, except for the species, 
because that is exactly what we want our model to predict.
"""
X = iris_df.drop(['species'], axis=1)
X_scaled = StandardScaler().fit_transform(X)

"""
Data modeling
Since we may not know how many clusters exist in the data, 
we begin by fitting K-means and examining the inertia values ​​for different values ​​of k. 
To do this, we will define a function called `kmeans_inertia` that takes `num_clusters` and `x_vals` (i.e., `X_scaled`) 
as arguments and returns a list containing the inertia corresponding to each value of k.
"""
num_clusters = [i for i in range(2, 11)]

def kmeans_inertia(num_clusters, x_vals):
    inertia = []
    for num in num_clusters:
        kms = KMeans(n_clusters= num, random_state= 42).fit(x_vals)
        inertia.append(kms.inertia_)
    return inertia

inertia = kmeans_inertia(num_clusters, X_scaled)
print(inertia)

"""
We will visualize the inertia data.
"""
plot = sns.lineplot(x=num_clusters, y=inertia, marker='o')
plot.set_xlabel('Number of clusters');
plot.set_ylabel('Inertia');
plt.savefig(iris_folder / 'inertia_plot.png', dpi=300)
plt.show()
"""
The plot seems to depict an elbow at four clusters, 
but there isn't a clear method for confirming that a four-cluster model is optimal. 
Therefore, the silhouette scores should be checked.
"""

"""
Results and evaluation

Now, we evaluate the silhouette score using the silhouette_score() function. 
Silhouette scores are used to study the distance between clusters.
Then, we compare the silhouette score of each value of k, 
from 2 through 10. 
To do this, we write a function called kmeans_sil that takes in num_clusters and x_vals (X_scaled)
 and returns a list of each k-value's silhouette score.
"""
def kmeans_sil(num_clusters, x_vals):
    sil_score = []
    for num in num_clusters:
        kms = KMeans(n_clusters= num, random_state= 42).fit(x_vals)
        sil_score.append(silhouette_score(x_vals, kms.labels_))
    return sil_score

sil_score = kmeans_sil(num_clusters, X_scaled)
print(sil_score)

"""
Next, we create a line plot that shows the relationship between num_clusters and sil_score. 
we use either seaborn or matplotlib to visualize this relationship.
"""
plot = sns.lineplot(x=num_clusters, y=sil_score, marker='o')
plot.set_xlabel('Num of clusters');
plot.set_ylabel('Silhouette Score');
plt.savefig(iris_folder / 'silhouette_plot.png', dpi=300)
plt.show()

"""
In the Silhouette Score plot we just uploaded, 
the highest point is actually at k = 2 (at 0.58 on the far left), 
while k = 4 is near 0.385 and marks the change in slope (the "elbow") where the decline begins to level off.

k = 2 is the absolute maximum (0.58): 
This indicates that splitting the data into 2 large groups produces the cleanest and most distinct separation.

k = 4 (0.385): Here, the line stops plummeting and begins to stabilize around 0.34, 
avoiding the sharp drop that occurs at k = 7 (0.266).

If we combine both plots:
k = 2 performs best in terms of silhouette (clusters that are well-separated, but very spread out internally).
k = 4 performs best according to the "elbow" method (clusters that are much more compact internally), 
while still maintaining a reasonable silhouette score (0.39).
"""
"""
Optimal K-value
"""
kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
print('Unique labels: ', np.unique(kmeans_4.labels_))

iris_df['clusters'] = kmeans_4.labels_
print(iris_df.head())

print(iris_df.groupby(by=['species', 'clusters']).size())
print(iris_df.groupby('clusters').mean(numeric_only=True))

"""
All of them are flowers of the *setosa* species,
but K-Means divided them into two subgroups based on the size of their sepals:
Cluster 1: *setosa* with wider and longer sepals (5.1 x 3.5 and 5.0 x 3.6).

Cluster 2: *setosa* with slightly narrower sepals (4.9 x 3.0, 4.7 x 3.2, 4.6 x 3.1).
"""
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=iris_df, x='clusters', hue='species', palette='viridis')
for container in ax.containers:
    ax.bar_label(container)

plt.title('Distribution of species by cluster (k=4)')
plt.xlabel('Assigned cluster')
plt.ylabel('Number of flowers')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig(iris_folder / 'Species_distribution.png', dpi=300)
plt.show()

print(iris_df.groupby(by=['clusters', 'species']).size().sort_values(ascending=False))
