#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# You are provided with a dataset containing various attributes of different wine samples. The goal of this assignment is to perform cluster analysis using the K-means algorithm to identify natural groupings in the data based on the attributes provided.
# 
# ## Dataset Overview
# The dataset consists of the following columns:
# 1. **Alcohol**: Alcohol content in the wine sample.
# 2. **Malic_Acid**: Amount of malic acid in the wine.
# 3. **Ash**: Ash content in the wine.
# 4. **Ash_Alcalinity**: Alkalinity of ash in the wine.
# 5. **Magnesium**: Magnesium content in the wine.
# 6. **Total_Phenols**: Total phenols content in the wine.
# 7. **Flavanoids**: Flavonoid content in the wine.
# 8. **Nonflavanoid_Phenols**: Non-flavonoid phenol content in the wine.
# 9. **Proanthocyanins**: Proanthocyanin content in the wine.
# 10. **Color_Intensity**: Intensity of the color of the wine.
# 11. **Hue**: Hue of the wine.
# 12. **OD280**: Ratio of OD280/OD315 of diluted wines.
# 13. **Proline**: Proline content in the wine.

# In[139]:


import pandas as pd


# In[141]:


df = pd.read_csv('WineData.csv')
df.head()


# In[143]:


df = df.drop('Unnamed: 0', axis=1)
df.head()


# ## Tasks
# 
# ### Task 1: Data Preprocessing
# - Handle any missing values if present.
# - Scale the data using `StandardScaler` or `MinMaxScaler` since K-means is sensitive to the scale of features.
# - Remove any unnecessary columns that don't contribute to clustering (e.g., index column if not relevant).

# In[146]:


df.info()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled.head()


# In[148]:


df_scaled.shape


# ### Task 2: Determine the Optimal Number of Clusters
# - Use the **Elbow method** to determine the optimal number of clusters.
# - Visualize the results using a line plot of the **Within-Cluster Sum of Squares (WCSS)** against the number of clusters.
# 
# 

# In[151]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.cluster import KMeans 
km = KMeans()
wcss = []
cluster = range(1,18)
for k in cluster:
    km1 = KMeans(n_clusters = k)
    km1.fit(df_scaled)
    wcss.append(km1.inertia_) 
print(wcss)
plt.plot(cluster,wcss,marker = 'o')


# In[152]:


from kneed import KneeLocator as kn
kl = kn(cluster,wcss,direction = 'decreasing',curve = 'convex')
kl.plot_knee()


# ### Task 3: K-means Clustering
# - Apply K-means clustering using the optimal number of clusters obtained from the Elbow method.
# - Assign cluster labels to each data point and create a new column in the dataset with these labels.
# 

# In[156]:


km2 = KMeans(n_clusters = 3)
km2.fit(df_scaled)
df_scaled['clust'] = km2.predict(df_scaled)
df_scaled.tail()


# In[ ]:





# ### Task 4: Cluster Analysis
# - Analyze the clusters by comparing the mean values of each feature within each cluster.
# - Visualize the clusters using a pairplot or scatterplot for selected features to understand the separations visually.
# 
# 

# In[159]:


df1 = df_scaled.drop('clust',axis = 1)
sns.pairplot(df1)


# In[160]:


cluster_analysis = df_scaled.groupby('clust').mean()
cluster_analysis


# ### Task 5: Interpretation
# - Interpret the characteristics of each cluster. For example, identify which cluster has the highest alcohol content, or which has the most intense color, etc.
# - Suggest potential names or categories for each cluster based on the observed characteristics.
# 

# In[161]:


for cluster in cluster_analysis.index:
    print(f"\nCluster {cluster} Characteristics:")
    for feature in cluster_analysis.columns:
        if cluster_analysis[feature][cluster] == cluster_analysis[feature].max():
            print(f"  - Highest {feature}")
        elif cluster_analysis[feature][cluster] == cluster_analysis[feature].min():
            print(f"  - Lowest {feature}")


# In[162]:


cluster_analysis.T.plot(kind='bar', figsize=(12, 6), colormap='cool')


# # Best of Luck
