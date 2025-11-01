import streamlit as st

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Urban Crime Analytics Visualization Dashboard",
    layout="wide"
)

# ------------------------------------------------------------
# Dashboard Introduction
# ------------------------------------------------------------
st.header("Explore trends, hotspots, and patterns through interactive visuals")

st.write(
    """
    This dashboard presents an interactive visualization of urban crime data, enabling a deeper understanding 
    of crime distribution and supporting data-driven decision-making for urban safety and policy planning.
    """
)

st.write(
    """
    The dataset, originally titled **"Uber and Urban Crime"** (published on 12 October 2019 by Bryan Weber),
    focuses primarily on urban crime data. Although the dataset references Uber, 
    the analysis in this dashboard emphasizes the crime dimension — exploring patterns, frequency, 
    and spatial distribution of criminal incidents within urban environments.
    """
)

# ------------------------------------------------------------
# Page Navigation Setup
# ------------------------------------------------------------
page1 = st.Page('Objectives1.py', title='Distribution and Correlation', icon=":material/bar_chart:")
page2 = st.Page('Objectives2.py', title='Group Comparisons and Chronotype', icon=":material/groups:")
page3 = st.Page('Objectives3.py', title='Preferred Start Time & Correlation Matrix', icon=":material/timeline:")

pg = st.navigation(
    {
        "Menu": [page1, page2, page3]
    }
)

pg.run()
