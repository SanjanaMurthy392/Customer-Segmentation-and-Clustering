#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/Lenovo/Downloads/Mall_Customers.csv")


# In[3]:


df.head()


# # Univariate Analysis

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# In[7]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[8]:


sns.kdeplot(data=df, x='Annual Income (k$)', shade=True, hue='Gender');


# In[9]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, shade=True, hue='Gender')
    plt.title(f'KDE Plot of {i} by Gender')
    plt.show();


# In[10]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df, x='Gender', y=df[i]);


# In[11]:


df['Gender'].value_counts()


# In[12]:


df['Gender'].value_counts(normalize=True)


# ## Bivariate Analysis

# In[13]:


sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[14]:


#df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[15]:


grouped_means = df.groupby('Gender')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

print(grouped_means)


# In[16]:


numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
correlation_matrix


# In[17]:


numeric_corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_corr, annot=True, cmap='coolwarm')
plt.show()


# # Clustering - Univariate, Bivariate, Multivariate

# In[18]:


clustering1 = KMeans(n_clusters=3)


# In[19]:


clustering1.fit(df[['Annual Income (k$)']])


# In[20]:


clustering1.labels_


# In[21]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[22]:


df['Income Cluster'].value_counts()


# In[23]:


clustering1.inertia_


# In[24]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[25]:


intertia_scores


# In[26]:


plt.plot(range(1,11),intertia_scores)


# In[27]:


df.columns


# In[29]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[30]:


#Bivariate Clustering


# In[31]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[32]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[33]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[34]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[35]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[37]:


print(df.columns)
result = df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(result)


# In[38]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[39]:


scale = StandardScaler()


# In[40]:


df.head()


# In[41]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[42]:


dff.columns


# In[43]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[44]:


dff = scale.fit_transform(dff)


# In[45]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[46]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[47]:


df


# In[48]:


df.to_csv('Clustering.csv')


# In[ ]:




