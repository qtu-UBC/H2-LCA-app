import streamlit as st
import pandas as pd
import os
from config.config import (
    IDEMAT_SHEET,
    H2_LCI_FOLDER,
    MAPPING_FILE,
    INPUTS_FILE,
    OUTPUTS_FILE,
    UNIQUE_FLOWS_FILE,
    GENERAL_INFO_FILE,
    OUTPUT_DIR,
)

# Read the CSV files
general_info_df = pd.read_csv(GENERAL_INFO_FILE)
inputs_df = pd.read_csv(INPUTS_FILE)
outputs_df = pd.read_csv(OUTPUTS_FILE)

# Set up the Streamlit page
st.title("H2 Manufacturing LCI Data Explorer")

# Create dropdown for Source_File selection
source_files = sorted(general_info_df['Source_File'].unique())
selected_source = st.selectbox(
    "Select a Source File:",
    options=source_files
)

st.markdown("### Filtering Inputs and Outputs Data")

if selected_source:
    # Filter inputs
    filtered_inputs = inputs_df[inputs_df['Source_File'] == selected_source]
    
    # Filter outputs 
    filtered_outputs = outputs_df[outputs_df['Source_File'] == selected_source]
    
    # Display inputs
    if not filtered_inputs.empty:
        display_df = filtered_inputs[['Flow', 'Category', 'Amount', 'Unit']]
        edited_inputs = st.data_editor(
            display_df,
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    help="Edit the amount value", 
                    step=0.01,
                )
            },
            hide_index=True,
            key="inputs_editor"
        )
        update_inputs = st.button("Update Input Values", key="update_inputs")
        
        if update_inputs:
            st.success("Input values updated successfully!")

        # Save the final values of input data
        saved_df_inputs = edited_inputs.copy()

        # print the saved_df_inputs
        # st.write(saved_df_inputs)

    else:
        st.info("No input data available for the selected source file.")

    # Add some vertical spacing
    st.markdown("<br><br>", unsafe_allow_html=True)
        
    # Display outputs    
    if not filtered_outputs.empty:
        display_df = filtered_outputs[['Flow', 'Category', 'Amount', 'Unit']]
        edited_outputs = st.data_editor(
            display_df,
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    help="Edit the amount value",
                    step=0.01,
                )
            },
            hide_index=True,
            key="outputs_editor"
        )
        update_outputs = st.button("Update Output Values", key="update_outputs")
        
        if update_outputs:
            st.success("Output values updated successfully!")

        # Save the final values of output data
        saved_df_outputs = edited_outputs.copy()

        # print the saved_df_outputs
        # st.write(saved_df_outputs)

    else:
        st.info("No output data available for the selected source file.")


st.markdown("### Mapping Inputs and Outputs to Idemat LCI Database")

# Import mapping function
from utils.mapper import map_flows

# Map both inputs and outputs to Idemat database
for df_name, df in [('inputs', saved_df_inputs), ('outputs', saved_df_outputs)]:
    if 'df' in locals() and not df.empty:
        flow_mappings = map_flows(df, MAPPING_FILE)
        st.write(f"\nMapped {df_name} flows:")
        st.write(flow_mappings)
        # Create a DataFrame to display mappings in a more readable format
        mapping_df = pd.DataFrame(list(flow_mappings.items()), columns=['Original Flow', 'Mapped Flow'])
        
        # Display the mapping results in a table
        st.markdown(f"#### {df_name.title()} Flow Mapping Results")
        st.dataframe(
            mapping_df,
            column_config={
                "Original Flow": st.column_config.Column("Original Flow", help="Flow name from input data"),
                "Mapped Flow": st.column_config.Column("Mapped Flow", help="Corresponding flow in Idemat database")
            },
            hide_index=True
        )
    else:
        st.info(f"No {df_name} data available for mapping.")
