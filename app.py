import streamlit as st
import pandas as pd
import os

# Read the CSV files
general_info_df = pd.read_csv("./output/general_information.csv")
inputs_df = pd.read_csv("./output/inputs.csv")

# Set up the Streamlit page
st.title("H2 Manufacturing LCI Data Explorer")

# Create dropdown for Source_File selection
source_files = sorted(general_info_df['Source_File'].unique())
selected_source = st.selectbox(
    "Select a Source File:",
    options=source_files
)

# Filter and display inputs data based on selection
if selected_source:
    filtered_inputs = inputs_df[inputs_df['Source_File'] == selected_source]
    
    if not filtered_inputs.empty:
        st.subheader(f"Inputs for {selected_source}")
        
        # Display selected columns
        display_df = filtered_inputs[['Flow', 'Category', 'Amount', 'Unit']]
        st.dataframe(display_df)
    else:
        st.info("No input data available for the selected source file.")
