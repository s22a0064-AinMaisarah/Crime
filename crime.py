import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Urban Crime Analytics Dashboard", page_icon="📊", layout="wide")

# --- HEADER ---
st.title("Urban Crime Analytics Dashboard 🏙️")
st.markdown("""
Using machine learning (K-Means & PCA) to uncover urban crime behavior patterns.
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
CSV_URL = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
ENCODING_TYPE = "cp1252"

df = load_data(CSV_URL, ENCODING_TYPE)

# --- MAIN APP ---
if not df.empty:

    # === INTRO SECTION ===
    st.subheader("Objective")
    st.write("""
    The objective of using K-Means clustering is to group cities into distinct clusters 
    based on their crime profiles — including violent, property, white-collar, and social crimes.  
    This enables city-level comparison and supports data-driven crime prevention strategies.
    """)
    st.markdown("---")

    # === SUMMARY STATISTICS ===
    st.subheader("Crime Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    num_cities = df["city_cat"].nunique() if "city_cat" in df.columns else len(df)
    avg_violent = df["violent_crime"].mean().round(2)
    avg_property = df["property_crime"].mean().round(2)
    avg_whitecollar = df["whitecollar_crime"].mean().round(2)
    avg_social = df["social_crime"].mean().round(2)

    col1.metric("Cities Analyzed", num_cities)
    col2.metric("Avg Violent Crime", avg_violent)
    col3.metric("Avg Property Crime", avg_property)
    col4.metric("Avg White-Collar Crime", avg_whitecollar)

    col5, col6 = st.columns(2)
    col5.metric("Avg Social Crime", avg_social)
    col6.metric("Clusters Formed", "3 (K-Means)")

    st.markdown("---")

    # === FEATURES & SCALING ===
    features = ["violent_crime", "property_crime", "whitecollar_crime", "social_crime"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # --- 1. ELBOW METHOD ---
    st.header("1. Determine Optimal Clusters (Elbow Method)")
    st.write("Use the Elbow Method to find the best number of clusters (k).")

    @st.cache_data
    def calculate_wcss(scaled_data):
        wcss = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        return wcss

    wcss = calculate_wcss(X_scaled)
    df_elbow = pd.DataFrame({"Number of Clusters (k)": range(2, 10), "WCSS": wcss})

    fig_elbow = px.line(
        df_elbow,
        x="Number of Clusters (k)",
        y="WCSS",
        title="Elbow Method for Optimal k",
        markers=True,
        template="plotly_white"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # --- 2. PCA CLUSTER VISUALIZATION ---
    st.header("2. Cluster Visualization (PCA)")
    k_clusters = st.slider("Select number of clusters (k):", 2, 9, 3)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df["crime_cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    df["PC1"], df["PC2"] = pca_data[:, 0], pca_data[:, 1]

    fig_pca = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="crime_cluster",
        hover_data=features,
        title=f"Crime Pattern Clusters (k={k_clusters})",
        template="plotly_white"
    )
    fig_pca.update_traces(marker=dict(size=10, opacity=0.8))
    st.plotly_chart(fig_pca, use_container_width=True)

    # --- 3. CLUSTER PROFILE ANALYSIS ---
    st.header("3. Cluster Profiles Analysis")
    st.write("Explore the average crime scores per cluster to understand key differences.")

    cluster_profile = df.groupby("crime_cluster")[features].mean().reset_index()
    st.dataframe(cluster_profile.style.background_gradient(cmap="viridis", axis=1, subset=features))

    cluster_profile_long = cluster_profile.melt(
        id_vars="crime_cluster",
        var_name="Crime Type",
        value_name="Average Score"
    )

    fig_profile = px.bar(
        cluster_profile_long,
        x="crime_cluster",
        y="Average Score",
        color="Crime Type",
        barmode="group",
        title="Average Crime Scores per Cluster",
        template="plotly_white"
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    # --- 4. INTERPRETATION ---
    st.markdown("---")
    st.subheader("Interpretation & Insights")
    st.success("""
    - The **Elbow Method** indicates **k=3** as the optimal cluster count.  
    - The **PCA scatter plot** shows clear separation between city clusters.  
    - Cluster insights reveal:
        - Cluster 1: Cities with **higher violent & property crimes**  
        - Cluster 2: Cities dominated by **white-collar crimes**  
        - Cluster 3: Cities showing **moderate or mixed patterns**
    - These insights can guide:
        ✅ Targeted policing strategies  
        ✅ Crime prevention campaigns  
        ✅ Urban safety policy planning
    """)

else:
    st.error("Failed to load dataset. Please check your GitHub data source.")
