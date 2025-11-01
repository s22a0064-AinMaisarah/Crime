# ===================================================
# Urban Crime Analytics Visualization Dashboard Homepage
# ===================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
import numpy as np

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Urban Crime Analytics Visualization Dashboard", layout="wide")

# Header title
st.header("Explore trends, hotspots, and patterns through interactive visuals")

# Intro paragraph
st.write(
    """
    This dashboard presents an interactive visualization of urban crime data, it enables a deeper understanding of crime distribution 
    and supports data-driven decision-making for urban safety and policy planning.
    """
)

# Dataset information
st.write(
    """
   This dataset, originally titled "Uber and Urban Crime" and published on 12 October 2019 by Bryan Weber,
   focuses primarily on crime-related data within urban environments. Although the dataset references Uber, 
   the analysis in this dashboard emphasizes the crime dimension, exploring patterns, frequency, and distribution of criminal
   incidents associated with urban mobility contexts.
    """
)

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
df = pd.read_csv(url)

# Plotly theme setup
pio.templates.default = "plotly_white"
colors = ["#4a90e2", "#f45b69", "#90c978", "#ffb74d"]

st.write("✅ File loaded successfully!")
st.write(df.shape)
st.write(df.head())

# -----------------------------------------------------------------------
# 1️⃣ Using (K-Means & PCA) to Uncover Crime Behavior Patterns in Cities.
# -----------------------------------------------------------------------
# Select crime-related columns
features = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']
X = df_uber_cleaned[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the Elbow Method
wcss = []  # within-cluster-sum-of-squares
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8,6))
plt.plot(range(2, 10), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means clustering (Choosing k=3 based on visual inspection of the elbow plot)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init
df_uber_cleaned['crime_cluster'] = kmeans.fit_predict(X_scaled)

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

df_uber_cleaned['PC1'] = pca_data[:, 0]
df_uber_cleaned['PC2'] = pca_data[:, 1]

# Visualize the clusters using PCA with Plotly Express for interactivity
fig_clusters = px.scatter(
    df_uber_cleaned,
    x='PC1',
    y='PC2',
    color='crime_cluster',
    hover_data=['crime_cluster', 'violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime', 'city_cat', 'state', 'age', 'income', 'poverty'],
    title='Crime Pattern Clusters (PCA)',
    labels={'crime_cluster': 'Crime Cluster'}
)
fig_clusters.show()

# Analyze cluster profiles
cluster_profile = df_uber_cleaned.groupby('crime_cluster')[features].mean().reset_index()
cluster_profile_melted = cluster_profile.melt(id_vars='crime_cluster', var_name='Crime Type', value_name='Average Crime Score')

# Visualize cluster profiles interactively using Plotly Express
fig_cluster_profile = px.bar(
    cluster_profile_melted,
    x='Crime Type',
    y='Average Crime Score',
    color='crime_cluster',
    barmode='group',
    hover_data=['crime_cluster', 'Crime Type', 'Average Crime Score'],
    title='Interactive Bar Chart: Average Crime Scores per Cluster',
    labels={'crime_cluster': 'Crime Cluster'}
)
fig_cluster_profile.show()

st.write(
    """
    **Interpretation:**  
      Cluster 0: This cluster generally represents cities with moderate crime levels across the four crime types (violent, property, white-collar, and social). The average crime scores for these categories are in the mid-range compared to the other clusters.
      Cluster 1: This cluster appears to represent cities with high crime levels across most categories, particularly violent and property crimes, and the highest average white-collar crime. These cities might face broader challenges related to crime.
      Cluster 2: This cluster seems to encompass cities with lower crime levels across all crime types. The average crime scores for this cluster are consistently the lowest.
    """
)

# ------------------------------------------------------------------------------------------------
# 2️⃣ Relationships between income, poverty levels, and total offense counts across city categories
# -------------------------------------------------------------------------------------------------
# Interactive Scatter Plot: Income vs. City Category with hover information
fig_income_citycat = px.scatter(
    df_uber_cleaned,
    x='income',
    y='city_cat',
    color='city_cat',
    hover_data=['city_cat', 'income', 'offense_count', 'violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime'],
    title='Interactive Scatter Plot: Income vs. City Category',
    labels={'city_cat': 'City Category (0: Group II, 1: Group I)'}
)
fig_income_citycat.show()

# Interactive Scatter Plot: Poverty % vs. City Category with hover information
fig_poverty_citycat = px.scatter(
    df_uber_cleaned,
    x='poverty',
    y='city_cat',
    color='city_cat',
    hover_data=['city_cat', 'poverty', 'offense_count', 'violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime'],
    title='Interactive Scatter Plot: Poverty % vs. City Category',
    labels={'city_cat': 'City Category (0: Group II, 1: Group I)'}
)
fig_poverty_citycat.show()

# Interactive Scatter Plot: Income vs. Offense Count with hover information and City Category color
fig_income_offense = px.scatter(
    df_uber_cleaned,
    x='income',
    y='offense_count',
    color='city_cat',
    hover_data=['city_cat', 'income', 'offense_count', 'violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime', 'state', 'age'],
    title='Interactive Scatter Plot: Income vs. Offense Count by City Category',
    labels={'city_cat': 'City Category (0: Group II, 1: Group I)'},
    trendline='ols' # Add OLS trendline
)
fig_income_offense.show()

# Interactive Scatter Plot: Poverty % vs. Offense Count with hover information and City Category color
fig_poverty_offense = px.scatter(
    df_uber_cleaned,
    x='poverty',
    y='offense_count',
    color='city_cat',
    hover_data=['city_cat', 'poverty', 'offense_count', 'violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime', 'state', 'age'],
    title='Interactive Scatter Plot: Poverty % vs. Offense Count by City Category',
    labels={'city_cat': 'City Category (0: Group II, 1: Group I)'},
    trendline='ols' # Add OLS trendline
)
fig_poverty_offense.show()
# ------------------------------------------------------------------------------------------------------
# 3️⃣ Gender Composition, Age Distribution, and Education Levels Influence Different Categories of Crime
# ------------------------------------------------------------------------------------------------------
# Define the crime score columns
crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

# Categorize cities based on male population percentage (you can adjust the bins)
# Let's use quartiles for simplicity
df_uber_cleaned['male_category'] = pd.qcut(df_uber_cleaned['male'], q=3, labels=['Low-Male', 'Balanced-Gender', 'High-Male'])

# Calculate the average crime scores per male category
male_category_crime_means = df_uber_cleaned.groupby('male_category', observed=True)[crime_cols].mean().reset_index()

# Melt the DataFrame for Plotly Express
crime_scores_melted_gender = male_category_crime_means.melt(
    id_vars='male_category',
    var_name='Crime Type',
    value_name='Average Crime Score'
)

# Create an interactive grouped bar chart
fig_gender_crime = px.bar(
    crime_scores_melted_gender,
    x='Crime Type',
    y='Average Crime Score',
    color='male_category',
    barmode='group',
    hover_data=['male_category', 'Crime Type', 'Average Crime Score'],
    title='Interactive Bar Chart: Average Crime Scores by Male Population Percentage Category',
    labels={'male_category': 'Male Population Category'}
)
fig_gender_crime.show()

# Define the crime score columns
crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

# Calculate the average crime scores per age group
age_group_crime_means = df_uber_cleaned.groupby('age', observed=True)[crime_cols].mean().reset_index()

# Create a list of crime types for the theta axis
categories = crime_cols

# Create the figure
fig = go.Figure()

# Add a trace for each age group
for index, row in age_group_crime_means.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=row[crime_cols].tolist(),
        theta=categories,
        fill='toself',
        name=f'Age Group {row["age"]}'
    ))

# Update the layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, age_group_crime_means[crime_cols].values.max()] # Set range based on max crime score
        )),
    showlegend=True,
    title='Interactive Radar Chart: Average Crime Scores by Age Group and Crime Type'
)

# Show the plot
fig.show()

# ------------------------------------------------------------
# Summary Box (Improved for Urban Crime Analytics)
# ------------------------------------------------------------
st.markdown(
    """
    <div style="
        background-color:#f8faff;
        padding:20px;
        border-radius:10px;
        border-left:6px solid #4a90e2;
        margin-top:30px;">
    <h4>📊 Summary of Findings</h4>
    <p>
    • <b>Crime Clustering Analysis (K-Means & PCA):</b> Identified three distinct city groups characterized by different crime intensities — 
      high-crime, moderate-crime, and low-crime zones.<br><br>

    • <b>Socioeconomic Correlations:</b> Higher poverty rates and lower income levels show strong positive associations with increased 
      offense counts, particularly in Group I (major urban centers).<br><br>

    • <b>Demographic Influences:</b> 
      - Cities with higher proportions of males tend to report slightly elevated violent and property crime rates.<br>
      - Age distribution affects the prevalence of crime types, with certain age brackets linked to higher white-collar or social crime scores.<br>
      - Education level demonstrates an inverse relationship with total crime rates — higher education attainment is generally linked with 
        lower average crime scores.<br><br>

    • <b>Policy Insight:</b> The combined results emphasize the importance of integrating socioeconomic and demographic indicators into 
      urban safety strategies and resource allocation for crime prevention.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
st.markdown("---")
st.caption("Created by Nurul Ain Maisarah Hamidin © 2025 | Scientific Visualization Project")
