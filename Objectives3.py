import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


st.title("Objective 3 — Preferred Start Time & Correlation Matrix")

# Load dataset
url = "https://raw.githubusercontent.com/nadiashahzanani/Sleep-Anxiety-Visualization/refs/heads/main/Time_to_think_Norburyy.csv"
df = pd.read_csv(url)

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
