import streamlit as st
import pandas as pd
import os
from config.config import (
    IDEMAT_SHEET,
    H2_LCI_FOLDER,
    MAPPING_FILE,
    INPUTS_FILE,
    OUTPUTS_FILE,
    UNIQUE_FLOWS_PROVIDERS_FILE,
    GENERAL_INFO_FILE,
    OUTPUT_DIR,
    RECIPE_CF_FILE,
)

# Read the CSV files
general_info_df = pd.read_csv(GENERAL_INFO_FILE)
inputs_df = pd.read_csv(INPUTS_FILE)
outputs_df = pd.read_csv(OUTPUTS_FILE)

# Set up the Streamlit page with full width
st.set_page_config(layout="wide")
st.title("H2 Manufacturing LCI Data Explorer")

# Create two columns with equal width
left_col, right_col = st.columns(2)

with left_col:
    # Create dropdown for Source_File selection
    source_files = sorted(general_info_df['Source_File'].unique())
    selected_source = st.selectbox(
        "Select a Source File:",
        options=source_files
    )

    st.markdown("### Input Data")

    if selected_source:
        # Filter and display inputs
        filtered_inputs = inputs_df[inputs_df['Source_File'] == selected_source]
        
        if not filtered_inputs.empty:
            display_df = filtered_inputs[['Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']]
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
                key="inputs_editor",
                use_container_width=True
            )
            update_inputs = st.button("Update Input Values", key="update_inputs")
            
            if update_inputs:
                st.success("Input values updated successfully!")

            saved_df_inputs = edited_inputs.copy()
        else:
            st.info("No input data available for the selected source file.")

    st.markdown("### Output Data")
    
    if selected_source:
        # Filter and display outputs
        filtered_outputs = outputs_df[outputs_df['Source_File'] == selected_source]
        
        if not filtered_outputs.empty:
            display_df = filtered_outputs[['Is reference?','Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']]
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
                key="outputs_editor",
                use_container_width=True
            )
            update_outputs = st.button("Update Output Values", key="update_outputs")
            
            if update_outputs:
                st.success("Output values updated successfully!")

            saved_df_outputs = edited_outputs.copy()
        else:
            st.info("No output data available for the selected source file.")

    st.markdown("### Flow Mapping")

    # Import mapping function
    from utils.mapper import map_flows

    # Initialize mapping DataFrames
    inputs_mapping_df = pd.DataFrame()
    outputs_mapping_df = pd.DataFrame()

    # Map both inputs and outputs to database
    try:
        for df_name, df in [('inputs', saved_df_inputs), ('outputs', saved_df_outputs)]:
            if 'df' in locals() and not df.empty:
                flow_mappings = map_flows(df, MAPPING_FILE)
                
                # Create mapping DataFrame
                mapping_records = []
                for orig_flow, mapping in flow_mappings.items():
                    record = {
                        'Original Flow': orig_flow,
                        'Mapped Flow': mapping['mapped_flow'],
                        'Amount': mapping['amount'],
                        'Unit': mapping['unit'],
                        'Category': mapping['category']
                    }
                    if 'is_reference' in mapping:
                        record['Is reference?'] = mapping['is_reference']
                    mapping_records.append(record)
                mapping_df = pd.DataFrame(mapping_records)
                
                # Save mapping DataFrame
                if df_name == 'inputs':
                    inputs_mapping_df = mapping_df
                else:
                    outputs_mapping_df = mapping_df
                
                # Display mapping results
                st.markdown(f"#### {df_name.title()} Mapping")
                st.dataframe(
                    mapping_df,
                    column_config={
                        "Original Flow": st.column_config.Column("Original Flow", help="Flow name from input data"),
                        "Mapped Flow": st.column_config.Column("Mapped Flow", help="Corresponding flow in database"),
                        "Amount": st.column_config.NumberColumn("Amount", help="Quantity", format="%.2f"),
                        "Unit": st.column_config.Column("Unit", help="Unit of measurement"),
                        "Category": st.column_config.Column("Category", help="Category of the flow")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info(f"No {df_name} data available for mapping.")
                
    except Exception as e:
        st.error(f"Error processing mapping: {str(e)}")

with right_col:
    # Import calculation function 
    from utils.calculate import calculate_impacts
    from utils.visualize import generate_impact_piechart

    st.markdown("### Impact Results")

    # Calculate and display inputs impacts
    if not inputs_mapping_df.empty:
        column_of_interest = "Carbon footprint (kg CO2 equiv.)"
        inputs_results_df = calculate_impacts(inputs_mapping_df, IDEMAT_SHEET, column_of_interest)
        
        if not inputs_results_df.empty:
            st.markdown("#### Inputs Impact")
            st.dataframe(
                inputs_results_df,
                column_config={
                    "Mapped Flow": st.column_config.Column("Mapped Flow", help="Flow in database"),
                    "Calculated Result": st.column_config.NumberColumn(
                        "CO2 Footprint (kg CO2)", 
                        help="Calculated impact",
                        format="%.3f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            inputs_total_impact = inputs_results_df['Calculated Result'].sum()
            st.metric(
                label="Total Input CO2 Impact", 
                value=f"{inputs_total_impact:.3f} kg CO2"
            )
        else:
            st.warning("No impact results for inputs")
    else:
        st.info("No input mapping data available")

    # Calculate and display outputs impacts
    if not outputs_mapping_df.empty:
        column_of_interest = "Carbon footprint (kg CO2 equiv.)"
        outputs_results_df = calculate_impacts(outputs_mapping_df, IDEMAT_SHEET, column_of_interest)
        
        if not outputs_results_df.empty:
            st.markdown("#### Outputs Impact")
            st.dataframe(
                outputs_results_df,
                column_config={
                    "Mapped Flow": st.column_config.Column("Mapped Flow", help="Flow in database"),
                    "Calculated Result": st.column_config.NumberColumn(
                        "CO2 Footprint (kg CO2)", 
                        help="Calculated impact",
                        format="%.3f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            outputs_total_impact = outputs_results_df['Calculated Result'].sum()
            st.metric(
                label="Total Output CO2 Impact",
                value=f"{outputs_total_impact:.3f} kg CO2"
            )
        else:
            st.warning("No impact results for outputs")
    else:
        st.info("No output mapping data available")

    st.markdown("### Visualization")
    
    chart_type = st.selectbox(
        "Select chart type",
        ["Pie Chart", "Bar Chart", "Line Chart"]
    )

    if st.button("Generate Chart"):
        if chart_type == "Pie Chart":
            pie_chart = generate_impact_piechart(inputs_results_df)
            st.pyplot(pie_chart)
        elif chart_type == "Bar Chart":
            st.info("Bar chart coming soon!")
        elif chart_type == "Line Chart": 
            st.info("Line chart coming soon!")
