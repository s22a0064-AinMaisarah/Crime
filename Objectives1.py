# ===================================================
# Sleep, Anxiety & Start Time Visualization Homepage
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
st.set_page_config(page_title="Sleep, Anxiety & Start Time Visualization", layout="wide")

# Header title
st.header("🧠 Time to Think — Sleep, Anxiety and University Start Time")

# Intro paragraph
st.write(
    """
    A scientific visualization exploring how **sleep quality**, **trait anxiety**, and **preferred class start times** interact among university students.
    """
)

# Dataset information
st.write(
    """
    This dashboard visualizes data from **Norbury & Evans (2018)** published in *Mendeley Data (V1)*.  
    The study explores psychological and behavioral patterns related to sleep, chronotype, and academic start times.
    """
)

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/nadiashahzanani/Sleep-Anxiety-Visualization/refs/heads/main/Time_to_think_Norburyy.csv"
df = pd.read_csv(url)

# Plotly theme setup
pio.templates.default = "plotly_white"
colors = ["#4a90e2", "#f45b69", "#90c978", "#ffb74d"]

st.write("✅ File loaded successfully!")
st.write(df.shape)
st.write(df.head())

# ------------------------------------------------------------
# 1️⃣ Sleep Quality Distribution (Histogram)
# ------------------------------------------------------------
fig1 = px.histogram(
    df, 
    x="psqi_2_groups", 
    nbins=15, 
    color_discrete_sequence=[colors[0]],
    title="1️⃣ Distribution of Sleep Quality (psqi_2_groups)"
)
fig1.update_layout(xaxis_title="PSQI (Higher = Poorer Sleep)", yaxis_title="Count")
st.plotly_chart(fig1, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    Sleep quality among students varies widely, with a notable portion reporting **poor or very poor sleep**.  
    This reflects the growing concern about sleep deprivation and its psychological effects among young adults.
    """
)

# ------------------------------------------------------------
# 2️⃣ Trait Anxiety Distribution (Histogram)
# ------------------------------------------------------------
fig2 = px.histogram(
    df, 
    x="Trait_Anxiety", 
    nbins=15, 
    color_discrete_sequence=[colors[1]],
    title="2️⃣ Distribution of Trait Anxiety Scores"
)
fig2.update_layout(xaxis_title="Trait Anxiety Score", yaxis_title="Frequency")
st.plotly_chart(fig2, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    Anxiety scores are distributed continuously across the student sample, 
    suggesting diverse levels of emotional resilience and stress management within the cohort.
    """
)

# ------------------------------------------------------------
# 3️⃣ Correlation Between Sleep Quality and Anxiety (Scatter)
# ------------------------------------------------------------
r, p = stats.pearsonr(df["psqi_2_groups"].dropna(), df["Trait_Anxiety"].dropna())
fig3 = px.scatter(
    df,
    x="psqi_2_groups",
    y="Trait_Anxiety",
    color_discrete_sequence=[colors[2]],
    trendline="ols",
    title=f"3️⃣ Relationship Between Sleep Quality and Anxiety (r = {r:.2f}, p = {p:.3g})"
)
fig3.update_layout(
    xaxis_title="Sleep Quality (Higher = Poorer Sleep)",
    yaxis_title="Trait Anxiety Score"
)
st.plotly_chart(fig3, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    The scatterplot reveals a **positive correlation** between poor sleep and higher anxiety levels.  
    Students with higher PSQI scores (indicating worse sleep) tend to report higher anxiety, reinforcing prior findings by Norbury & Evans (2018).
    """
)

# ------------------------------------------------------------
# 4️⃣ Trait Anxiety by Sleep Category (Box Plot)
# ------------------------------------------------------------
fig4 = px.box(
    df,
    x="Sleep_Category",
    y="Trait_Anxiety",
    color="Sleep_Category",
    color_discrete_sequence=[colors[0], colors[1]],
    title="4️⃣ Trait Anxiety by Sleep Category"
)
fig4.update_layout(xaxis_title="Sleep Category", yaxis_title="Anxiety Score")
st.plotly_chart(fig4, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    Students classified as *poor sleepers* exhibit **higher median anxiety levels** compared to those with good sleep quality.  
    This highlights the bidirectional link between psychological stress and sleep disturbance.
    """
)

# ------------------------------------------------------------
# 5️⃣ Chronotype and Start Time Preferences (Grouped Bar Chart)
# ------------------------------------------------------------
ctab = pd.crosstab(df["Start_time_code"], df["MEQ"])
ctab = ctab.reset_index().melt(id_vars="Start_time_code", var_name="Chronotype", value_name="Count")

fig5 = px.bar(
    ctab,
    x="Start_time_code",
    y="Count",
    color="Chronotype",
    color_discrete_sequence=colors,
    barmode="group",
    title="5️⃣ Preferred Start Time by Chronotype"
)
fig5.update_layout(xaxis_title="Preferred Start Time", yaxis_title="Count")
st.plotly_chart(fig5, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    Morning-type students generally prefer earlier class schedules, while evening-types lean toward later start times.  
    This reflects the influence of **chronotype** on daily energy patterns and academic engagement.
    """
)

# ------------------------------------------------------------
# 6️⃣ Correlation Heatmap (Matrix)
# ------------------------------------------------------------
numeric = df.select_dtypes(include=np.number)
corr = numeric.corr().round(2)

fig6 = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        reversescale=True
    )
)
fig6.update_layout(title="6️⃣ Correlation Matrix: Sleep, Anxiety & Behavioral Variables")
st.plotly_chart(fig6, use_container_width=True)

st.write(
    """
    **Interpretation:**  
    Strongest relationships appear between **PSQI ↔ Anxiety** and **PSQI ↔ Daytime Dozing**,  
    indicating that poorer sleep quality contributes to both **emotional stress** and **daytime fatigue**.
    """
)

# ------------------------------------------------------------
# Summary Box
# ------------------------------------------------------------
st.markdown(
    """
    <div style="
        background-color:#f8faff;
        padding:20px;
        border-radius:10px;
        border-left:6px solid #4a90e2;
        margin-top:30px;">
    <h4>📘 Summary</h4>
    <p>
    • Poorer sleep quality is strongly linked with higher trait anxiety.<br>
    • Chronotype influences preferred class start times (morning vs evening types).<br>
    • Daytime fatigue and anxiety co-occur among students with poor sleep.<br>
    • Findings replicate <b>Norbury & Evans (2018)</b> and emphasize the need for healthy sleep interventions in universities.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
st.markdown("---")
st.caption("Created by Nadia Shahzanani © 2025 | Scientific Visualization Project")
