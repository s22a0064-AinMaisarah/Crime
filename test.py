import streamlit as st
import pandas as pd

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
