import streamlit as st
import pandas as pd
import plotly.express as px
# Import streamlit for the final display

# Define the URL for the CSV file
CSV_URL = 'https://raw.githubusercontent.com/s22a0064-AinMaisarah/Crime/refs/heads/main/df_crime_cleaned.csv'

# Use st.cache_data to load the data only once and improve performance
# This function will only rerun if the URL (or the function code) changes
@st.cache_data
def load_data(url):
    """Reads the CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# --- Streamlit App Layout ---

st.title(' crime Data Viewer 🏙️')

df_uber = load_data(CSV_URL)

if not df_uber.empty:
    st.header('Loaded Data Preview')
    st.write(f'Successfully loaded {len(df_uber)} rows and {len(df_uber.columns)} columns.')
    
    # Display the DataFrame in an interactive table
    st.dataframe(df_uber)
    
    # Optionally, display some basic statistics
    st.subheader('Basic Data Info')
    st.text(df_uber.info(buf=None)) # Display info method output

else:
    st.warning("Data could not be loaded. Please check the URL or your connection.")



# Assuming df_uber_cleaned is your loaded DataFrame

# Select education and crime score columns
education_cols = ['high_school_below', 'high_school', 'some_college', 'bachelors_degree']
crime_cols = ['violent_crime', 'property_crime', 'whitecollar_crime', 'social_crime']

# --- Data Preparation (Remains the same as it uses pandas) ---

# Melt the DataFrame to a long format for plotting
# 1. Melt the crime columns
crime_melted = df_uber_cleaned.melt(
    value_vars=crime_cols,
    var_name='Crime Type',
    value_name='Crime Score',
    id_vars=education_cols
)

# 2. Melt the education columns within the crime melted data
education_crime_melted = crime_melted.melt(
    id_vars=['Crime Type', 'Crime Score'],
    value_vars=education_cols,
    var_name='Education Level',
    value_name='Education Percentage'
)

# --- Plotly Express Plotting ---

# Create the grouped violin plot using plotly.express
# violinmode='group' creates separate violins for each 'Crime Type' side-by-side within each 'Education Level'
# box=True adds a box plot inside the violin, similar to seaborn's inner='quartile'
fig = px.violin(
    education_crime_melted,
    x='Education Level',
    y='Crime Score',
    color='Crime Type',
    violinmode='group',
    box=True, # Adds a box plot inside the violin
    points=False, # Optional: Set to 'all' to show all data points
    title='Distribution of Crime Scores by Education Level and Crime Type'
)

# Customize layout (optional, but good for clarity)
fig.update_layout(
    xaxis_title='Education Level',
    yaxis_title='Crime Score',
    legend_title='Crime Type',
    # Adjusting the x-axis to ensure labels fit (Plotly handles rotation better than Matplotlib)
    xaxis={'categoryorder': 'array', 'categoryarray': education_cols}
)

# --- Streamlit Display ---

# In your Streamlit app, display the figure using st.plotly_chart
# st.plotly_chart(fig, use_container_width=True) # Recommended for Streamlit
