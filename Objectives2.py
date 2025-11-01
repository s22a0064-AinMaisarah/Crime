import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Objective 2 — Group Comparisons", layout="wide")
st.title("Objective 2 — Group Comparisons and Chronotype")

url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
df = pd.read_csv(url)
st.success("✅ Data loaded successfully!")

# Scatter plots
fig1 = px.scatter(df, x='income', y='offense_count', color='city_cat',
                  title='Income vs Offense Count by City Category',
                  trendline='ols')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df, x='poverty', y='offense_count', color='city_cat',
                  title='Poverty vs Offense Count by City Category',
                  trendline='ols')
st.plotly_chart(fig2, use_container_width=True)
