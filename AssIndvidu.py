import streamlit as st

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Urban Crime Analytics Visualization Dashboard", layout="wide")

# Header title
st.header("Explore trends, hotspots, and patterns through interactive visuals")

# Intro paragraph
st.write(
    """
    This dashboard presents an interactive visualization of urban crime data, enabling a deeper understanding 
    of crime distribution and supporting data-driven decision-making for urban safety and policy planning.
    """
)

# Dataset information
st.write(
    """
   This dataset, originally titled "Uber and Urban Crime" (published by Bryan Weber, 2019),
   focuses primarily on crime-related data within urban environments. The analysis emphasizes
   the crime dimension—exploring patterns, frequency, and distribution of incidents in cities.
    """
)

# Navigation Pages
page1 = st.Page("pages/Objectives1.py", title="Distribution and Correlation", icon=":material/bar_chart:")
page2 = st.Page("pages/Objectives2.py", title="Group Comparisons and Chronotype", icon=":material/groups:")
page3 = st.Page("pages/Objectives3.py", title="Preferred Start Time & Correlation Matrix", icon=":material/timeline:")

pg = st.navigation({"Menu": [page1, page2, page3]})
pg.run()
