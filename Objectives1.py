import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------------------------------------------
st.set_page_config(page_title="Objective 1 — Distribution and Correlation", layout="wide")

st.title("Objective 1 — Distribution and Correlation")

url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
df = pd.read_csv(url)

st.success("✅ File loaded successfully!")
st.write(df.shape)
st.dataframe(df.head())

features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(2, 10), wcss, marker='o')
ax.set_title('Elbow Method for Optimal k')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# K-Means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['crime_cluster'] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
df['PC1'], df['PC2'] = pca_data[:, 0], pca_data[:, 1]

fig_clusters = px.scatter(
    df,
    x='PC1', y='PC2',
    color='crime_cluster',
    hover_data=['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime', 'state', 'income', 'poverty'],
    title='Crime Pattern Clusters (PCA)',
)
st.plotly_chart(fig_clusters, use_container_width=True)

cluster_profile = df.groupby('crime_cluster')[features].mean().reset_index()
melted = cluster_profile.melt(id_vars='crime_cluster', var_name='Crime Type', value_name='Average Score')

fig_bar = px.bar(
    melted,
    x='Crime Type', y='Average Score', color='crime_cluster',
    barmode='group',
    title='Average Crime Scores per Cluster'
)
st.plotly_chart(fig_bar, use_container_width=True)
