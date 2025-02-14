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
    RECIPE_CF_FILE,
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

# Initialize mapping DataFrames
inputs_mapping_df = pd.DataFrame()
outputs_mapping_df = pd.DataFrame()

# Map both inputs and outputs to database
try:
    # Map both inputs and outputs
    for df_name, df in [('inputs', saved_df_inputs), ('outputs', saved_df_outputs)]:
        if 'df' in locals() and not df.empty:
            flow_mappings = map_flows(df, MAPPING_FILE)
            st.write(f"\nMapped {df_name} flows:")
            st.write(flow_mappings)
            
            # Create a DataFrame to display mappings in a more readable format
            mapping_records = []
            for orig_flow, mapping in flow_mappings.items():
                mapping_records.append({
                    'Original Flow': orig_flow,
                    'Mapped Flow': mapping['mapped_flow'],
                    'Amount': mapping['amount'],
                    'Unit': mapping['unit']
                })
            mapping_df = pd.DataFrame(mapping_records)
            
            # Save mapping DataFrame based on df_name
            if df_name == 'inputs':
                inputs_mapping_df = mapping_df
            else:
                outputs_mapping_df = mapping_df
            
            # Display the mapping results in a table
            st.markdown(f"#### {df_name.title()} Flow Mapping Results")
            st.dataframe(
                mapping_df,
                column_config={
                    "Original Flow": st.column_config.Column("Original Flow", help="Flow name from input data"),
                    "Mapped Flow": st.column_config.Column("Mapped Flow", help="Corresponding flow in combined database"),
                    "Amount": st.column_config.NumberColumn("Amount", help="Quantity of the flow", format="%.2f"),
                    "Unit": st.column_config.Column("Unit", help="Unit of measurement")
                },
                hide_index=True
            )
        else:
            st.info(f"No {df_name} data available for mapping.")
            
except Exception as e:
    st.error(f"Error processing mapping: {str(e)}")

# Import calculation function 
from utils.calculate import calculate_impacts

st.markdown("### Impact Calculation Results")

# Calculate impacts for inputs
if not inputs_mapping_df.empty:
    # specify the column of interest
    column_of_interest = "Carbon footprint (kg CO2 equiv.)"
    # Calculate impacts using CO2 footprint column
    inputs_results_df = calculate_impacts(inputs_mapping_df, IDEMAT_SHEET, column_of_interest)
    
    if not inputs_results_df.empty:
        st.markdown("#### Inputs Impact Results")
        st.dataframe(
            inputs_results_df,
            column_config={
                "Mapped Flow": st.column_config.Column("Mapped Flow", help="Flow in Idemat database"),
                "Calculated Result": st.column_config.NumberColumn(
                    "CO2 Footprint (kg CO2)", 
                    help="Calculated CO2 impact",
                    format="%.3f"
                )
            },
            hide_index=True
        )
        
        # Show total impact
        inputs_total_impact = inputs_results_df['Calculated Result'].sum()
        st.metric(
            label="Total CO2 Impact for Inputs", 
            value=f"{inputs_total_impact:.3f} kg CO2"
        )
    else:
        st.warning("No impact results calculated for inputs")
else:
    st.info("No mapping data available for inputs impact calculation")

# Calculate impacts for outputs
if not outputs_mapping_df.empty:
    # specify the column of interest
    column_of_interest = "Carbon footprint (kg CO2 equiv.)"
    # Calculate impacts using CO2 footprint column  
    outputs_results_df = calculate_impacts(outputs_mapping_df, IDEMAT_SHEET, column_of_interest)
    
    if not outputs_results_df.empty:
        st.markdown("#### Outputs Impact Results")
        st.dataframe(
            outputs_results_df,
            column_config={
                "Mapped Flow": st.column_config.Column("Mapped Flow", help="Flow in Idemat database"),
                "Calculated Result": st.column_config.NumberColumn(
                    "CO2 Footprint (kg CO2)", 
                    help="Calculated CO2 impact",
                    format="%.3f"
                )
            },
            hide_index=True
        )
        
        # Show total impact
        outputs_total_impact = outputs_results_df['Calculated Result'].sum()
        st.metric(
            label="Total CO2 Impact for Outputs",
            value=f"{outputs_total_impact:.3f} kg CO2"
        )
    else:
        st.warning("No impact results calculated for outputs")
else:
    st.info("No mapping data available for outputs impact calculation")
