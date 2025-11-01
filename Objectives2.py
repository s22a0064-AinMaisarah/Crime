import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


st.title("Objective 2 — Group Comparisons and Chronotype")

# Load dataset
url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
df = pd.read_csv(url)

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
