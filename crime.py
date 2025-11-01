import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Urban Crime Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- HEADER ---
st.title("Urban Crime Analytics Dashboard")
st.markdown("""
Using machine learning (K-Means & PCA) to uncover crime behavior patterns in cities.
""")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data(url, encoding):
    try:
        df = pd.read_csv(url, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- DATA URL ---
CSV_URL = 'https://raw.githubusercontent.com/s22a0064-AinMaisarah/UrbanCrime/refs/heads/main/df_uber_cleaned.csv'
ENCODING_TYPE = 'cp1252'

df = load_data(CSV_URL, ENCODING_TYPE)

# --- MAIN APP ---
if not df.empty:

    # === INTRO SECTION ===
    st.subheader("Objective")
    st.write("""
    The objective of using K-Means clustering is to group cities into three distinct clusters 
    based on their crime profiles including violent, property, white-collar, and social crimes, 
    so that cities with similar crime patterns are categorized together. 
    This allows for clear comparison between areas with different crime characteristics and supports targeted crime-prevention strategies.
    """)

    st.markdown("---")

    # === SUMMARY STATISTICS ===
    st.subheader("Crime Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    num_cities = df['city_cat'].nunique() if 'city_cat' in df.columns else len(df)
    avg_violent = df['violent_crime'].mean().round(2)
    avg_property = df['property_crime'].mean().round(2)
    avg_whitecollar = df['whitecollar_crime'].mean().round(2)
    avg_social = df['social_crime'].mean().round(2)

    col1.metric(
        label="Cities Analyzed",
        value=num_cities,
        help="Total number of unique cities analyzed in the dataset.",
        border=True,
    )
    col2.metric(
        label="Avg Violent Crime",
        value=avg_violent,
        help="Average number of violent crime incidents per city.",
        border=True,
    )
    col3.metric(
        label="Avg Property Crime",
        value=avg_property,
        help="Average number of property crime incidents per city.",
        border=True,
    )
    col4.metric(
        label="Avg White-Collar Crime",
        value=avg_whitecollar,
        help="Average number of white-collar crimes such as fraud and embezzlement per city.",
        border=True,
    )

    col5, col6 = st.columns(2)
    col5.metric(
        label="Avg Social Crime",
        value=avg_social,
        help="Average rate of social-related crimes (e.g., gambling, drug offenses).",
        border=True,
    )
    col6.metric(
        label="Clusters Formed",
        value="3 (K-Means)",
        help="Number of city clusters identified using K-Means.",
        border=True,
    )

    st.markdown("---")

   import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# Import Plotly for the conversion
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # Needed for the elbow method plot range

# --- Assuming 'df' is your loaded DataFrame from the previous step ---
# --- (Placeholder for context) ---
# df = pd.read_csv(...) 
# features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
# X = df[features]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# === MACHINE LEARNING PIPELINE ===
st.subheader("Machine Learning Workflow")

st.info("""
The data was standardized using **StandardScaler**, clustered with **K-Means (k=3)**,
and reduced to two principal components using **PCA** for visualization.
""")

# Note: Ensure 'df' and 'features' are defined before this code block runs
# Placeholder initialization for the conversion to work if run standalone
if 'df' not in locals():
    st.warning("Please ensure the 'df' DataFrame is loaded and 'features' are defined.")
    # Create dummy data for demonstration purposes only
    np.random.seed(42)
    df = pd.DataFrame({
        'violent_crime': np.random.rand(100)*100,
        'property_crime': np.random.rand(100)*150,
        'whitecollar_crime': np.random.rand(100)*50,
        'social_crime': np.random.rand(100)*75
    })
    features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


# --- Elbow Method (Plotly Conversion) ---
st.subheader("Elbow Method – Optimal Number of Clusters")
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Convert Matplotlib Elbow plot to Plotly
fig_elbow = go.Figure(data=go.Scatter(
    x=list(range(2, 10)), 
    y=wcss, 
    mode='lines+markers',
    marker=dict(size=8)
))
fig_elbow.update_layout(
    title="Elbow Method for Optimal k",
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Within-Cluster Sum of Squares (WCSS)",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_elbow, use_container_width=True)


# --- Apply KMeans ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['crime_cluster'] = kmeans.fit_predict(X_scaled).astype(str) # Convert cluster to string for Plotly hue


# --- PCA Visualization (Plotly Conversion) ---
st.subheader("PCA Visualization – Crime Pattern Clusters")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
df['PC1'], df['PC2'] = pca_data[:, 0], pca_data[:, 1]

# Convert Seaborn/Matplotlib Scatter plot to Plotly
fig_pca = px.scatter(
    df, 
    x='PC1', 
    y='PC2', 
    color='crime_cluster',
    title="Crime Pattern Clusters (PCA Projection)",
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'crime_cluster': 'Cluster'},
    color_discrete_sequence=px.colors.qualitative.Vivid, # Use a vibrant color palette
    hover_data=df.columns
)
fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig_pca.update_layout(height=500)
st.plotly_chart(fig_pca, use_container_width=True)


# --- Cluster Profile (Plotly Conversion) ---
st.subheader("Cluster Profile – Average Crime Rates per Cluster")
cluster_profile = df.groupby('crime_cluster')[features].mean().T

# Convert Matplotlib Bar plot to Plotly
fig_profile = px.bar(
    cluster_profile.reset_index(),
    x='index',
    y=cluster_profile.columns,
    barmode='group',
    title="Crime Type Distribution by Cluster",
    labels={'index': 'Crime Type', 'value': 'Average Crime Level', 'crime_cluster': 'Cluster'},
    color_discrete_sequence=px.colors.qualitative.Vivid
)
fig_profile.update_layout(
    xaxis_title="Crime Type", 
    yaxis_title="Average Crime Level",
    legend_title="Cluster",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_profile, use_container_width=True)
    # === INTERPRETATION ===
    st.markdown("---")
    st.subheader("Interpretation & Insights")
    st.success("""
    - The **Elbow Method** validates that **k=3** clusters provide the optimal segmentation.  
    - The **PCA plot** clearly shows three distinct clusters, reflecting strong separation in crime behavior.  
    - Cluster patterns suggest:
        - Some cities face **higher violent & property crimes**.
        - Others are dominated by **white-collar or social crimes**.
    - Insights from this model can guide:
        ✅ Targeted law enforcement deployment  
        ✅ Crime prevention strategies  
        ✅ Resource prioritization and urban safety planning  
    """)

else:
    st.error("Failed to load dataset. Please check the data source.")
