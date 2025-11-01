import streamlit as st
import pandas as pd
import numpy as np  # Needed for dummy data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Crime Cluster Analysis", layout="wide")
st.title("Crime Pattern Clustering Analysis 🏙️")
st.write("This app analyzes crime data to find distinct clusters using K-Means and visualizes the results interactively.")

# --- 1. Load and Prepare Data ---
# Since we don't have the original df_uber_cleaned, we'll create sample data.
# In your real app, you would load your 'df_uber_cleaned' here.
@st.cache_data
def load_data():
    """Creates synthetic data for demonstration."""
    np.random.seed(42)
    data = np.random.rand(200, 4) * 10
    # Create some differentiation for clustering
    data[:50, :2] += 10  # Group 1: High violent/property
    data[50:120, 2:] += 10 # Group 2: High whitecollar/social
    data[120:, 1:3] += 5   # Group 3: Mid property/whitecollar
    
    df = pd.DataFrame(data, columns=['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime'])
    return df

df_uber_cleaned = load_data()
features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

with st.expander("View Raw Sample Data"):
    st.dataframe(df_uber_cleaned.head())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_uber_cleaned[features])

# --- 2. Determine Optimal k (Elbow Method) ---
st.header("1. Determine Optimal Clusters (Elbow Method)")
st.write("The Elbow Method helps find the best number of clusters (k). The 'elbow' (point of diminishing returns) is the ideal k.")

@st.cache_data
def calculate_wcss(scaled_data):
    """Calculates WCSS for k from 2 to 9."""
    wcss = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    return wcss

wcss = calculate_wcss(X_scaled)

# Create a DataFrame for the Plotly plot
df_elbow = pd.DataFrame({
    'Number of Clusters (k)': range(2, 10),
    'WCSS': wcss
})

# Plot with Plotly Express (Replaces plt.plot)
fig_elbow = px.line(
    df_elbow,
    x='Number of Clusters (k)',
    y='WCSS',
    title='Elbow Method for Optimal k',
    markers=True,  # Adds markers just like marker='o'
    template='plotly_white'
)
fig_elbow.update_traces(marker=dict(size=8))
st.plotly_chart(fig_elbow, use_container_width=True)

# --- 3. Interactive Clustering and Visualization ---
st.header("2. Cluster Visualization (PCA)")
st.write("Select the number of clusters (k) based on the elbow plot above. The plot below shows the clusters in 2D using PCA.")

# Add a slider for user to choose k
k_clusters = st.slider('Select number of clusters (k):', min_value=2, max_value=9, value=3)

# Apply K-Means clustering with the selected k
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df_uber_cleaned['crime_cluster'] = kmeans.fit_predict(X_scaled)

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

df_uber_cleaned['PC1'] = pca_data[:, 0]
df_uber_cleaned['PC2'] = pca_data[:, 1]

# Add original features to the DataFrame for hover info
for feature in features:
    df_uber_cleaned[feature] = df_uber_cleaned[feature]

# Visualize the clusters using PCA with Plotly Express (Replaces sns.scatterplot)
fig_pca = px.scatter(
    df_uber_cleaned,
    x='PC1',
    y='PC2',
    color='crime_cluster',  # Use cluster labels for color
    title=f'Crime Pattern Clusters (k={k_clusters})',
    hover_data=features,    # Show original crime data on hover
    template='plotly_white'
)
fig_pca.update_traces(marker=dict(size=10, opacity=0.8))
st.plotly_chart(fig_pca, use_container_width=True)

# --- 4. Analyze Cluster Profiles ---
st.header("3. Cluster Profiles Analysis")
st.write("This shows the average crime score for each cluster, helping us understand what defines each group.")

# Analyze cluster profiles
cluster_profile = df_uber_cleaned.groupby('crime_cluster')[features].mean().reset_index()

# Display the profile table
st.subheader("Cluster Profile Averages (Mean Values)")
st.dataframe(cluster_profile.style.background_gradient(cmap='viridis', axis=1, subset=features))

# Visualize cluster profiles with Plotly Express (Replaces pandas .plot(kind='bar'))

# We need to "melt" the DataFrame to a long format for Plotly Express
cluster_profile_long = cluster_profile.melt(
    id_vars='crime_cluster',
    var_name='Crime Type',
    value_name='Average Score'
)

fig_profile = px.bar(
    cluster_profile_long,
    x='crime_cluster',
    y='Average Score',
    color='Crime Type',       # Creates grouped bars by crime type
    barmode='group',          # Explicitly sets to grouped mode
    title='Crime Type Averages per Cluster',
    template='plotly_white'
)
fig_profile.update_layout(xaxis_title='Crime Cluster')
st.plotly_chart(fig_profile, use_container_width=True)
