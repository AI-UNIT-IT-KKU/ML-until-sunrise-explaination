# Clustering — grouping unlabeled data

## What clustering is
Clustering is about discovering structure in unlabeled data. Instead of predicting a category, we try to group similar points together. You can think of it as asking the data, “who looks like who?” and letting the algorithm draw groups.

---

## K‑Means Clustering

### What it is
K‑Means partitions the dataset into *k* clusters. Each cluster is represented by its centroid, and each point is assigned to the cluster whose centroid is closest.

### The elbow method
The challenge is picking the right *k*. One common approach is the **elbow method**. We measure the **within‑cluster sum of squares (WCSS)** for different values of *k*. WCSS is basically the sum of squared distances from each point to its cluster centroid. As *k* increases, WCSS decreases. The “elbow” is where the rate of decrease sharply changes — that’s a good choice for *k*.

Formula:
\[ WCSS = \sum_{i=1}^k \sum_{x \in C_i} \lVert x - \mu_i \rVert^2 \]

Code:
```python
wcss = []
for i in range(1, 11):
    kmean = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
```

### Training K‑Means
Once you’ve chosen an optimal *k*, you fit the model and assign clusters:
```python
kmean = KMeans(n_clusters=optimal_value, init='k-means++', random_state=42)
y_kmean = kmean.fit_predict(X)
print(y_kmean)  # cluster labels for each point
```

### The random initialization trap
If centroids are chosen randomly at the start, the algorithm might get stuck in a poor local optimum. This is the **random initialization trap**. The fix is **k‑means++**, which improves the choice of initial centroids:
1. Pick the first centroid randomly.
2. For every data point, compute its distance to the nearest selected centroid.
3. Choose the next centroid with weighted probability: points farther away have a higher chance of being chosen.
4. Repeat steps 2–3 until all centroids are chosen.
5. Run the standard K‑Means algorithm.

This avoids bad starting points and usually leads to better clustering.

---

## Hierarchical Clustering

### Two flavors
- **Agglomerative**: bottom‑up. Start with each point as its own cluster, then repeatedly merge the closest clusters until only one remains.
- **Divisive**: top‑down. Start with all points in one cluster, then split recursively.

### Agglomerative approach in plain steps
1. Start: each data point is its own cluster.
2. Find the two closest clusters and merge them.
3. Repeat step 2 until there’s only one giant cluster.

The “closeness” depends on how you define **distance between clusters**:
- Minimum distance (closest points, single linkage)
- Maximum distance (farthest points, complete linkage)
- Average of all pairwise distances (average linkage)
- Distance between centroids (centroid linkage)

These choices matter, because they affect the final cluster structure.

### Dendrograms
A **dendrogram** is a tree diagram showing how clusters merge at each step. By looking at the distances at which clusters merge, you can decide on the optimal number of clusters.

Code:
```python
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
```
Here `ward` linkage minimizes the variance within clusters at each merge. Look for a big vertical jump in the dendrogram to decide where to “cut” the tree.

### Training Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=optimal_value, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
```

This groups the data bottom‑up, using Euclidean distance and Ward’s method, and outputs cluster labels for each point.

---

## Recap
- K‑Means is fast and works well when clusters are roughly spherical. Use k‑means++ and elbow method to get better results.
- Hierarchical clustering doesn’t need you to pre‑specify *k*. The dendrogram helps you decide it visually. Choice of linkage method makes a big difference.

Image placeholders you might add later:
- `assets/kmeans-elbow.png` (elbow method)
- `assets/kmeans-centroids.png` (showing centroids and clusters)
- `assets/dendrogram.png` (hierarchical clustering cut)

