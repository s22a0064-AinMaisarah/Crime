import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Objective 3 — Preferred Start Time & Correlation Matrix", layout="wide")
st.title("Objective 3 — Preferred Start Time & Correlation Matrix")

url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv"
df = pd.read_csv(url)
st.success("✅ Data loaded successfully!")

crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

# Gender-based crime visualization
df['male_category'] = pd.qcut(df['male'], q=3, labels=['Low-Male', 'Balanced', 'High-Male'])
male_means = df.groupby('male_category')[crime_cols].mean().reset_index()
melted = male_means.melt(id_vars='male_category', var_name='Crime Type', value_name='Average Score')

fig_bar = px.bar(melted, x='Crime Type', y='Average Score', color='male_category',
                 barmode='group', title='Crime by Male Population Category')
st.plotly_chart(fig_bar, use_container_width=True)

# Radar chart by age
age_means = df.groupby('age')[crime_cols].mean().reset_index()
categories = crime_cols
fig = go.Figure()
for _, row in age_means.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=row[crime_cols].values,
        theta=categories,
        fill='toself',
        name=f'Age {row["age"]}'
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title='Average Crime Scores by Age Group',
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)
