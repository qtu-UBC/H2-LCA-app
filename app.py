import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path

# Try to import reportlab at module level
try:
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    REPORTLAB_AVAILABLE = False
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

# Add scripts directory to path for importing the processor
sys.path.append(str(Path(__file__).parent / "scripts"))

# Read the CSV files
general_info_df = pd.read_csv(GENERAL_INFO_FILE)
inputs_df = pd.read_csv(INPUTS_FILE)
outputs_df = pd.read_csv(OUTPUTS_FILE)
unique_locations_df = pd.read_csv(UNIQUE_FLOWS_PROVIDERS_FILE)

# Set up the Streamlit page with full width
st.set_page_config(layout="wide")

# Title
st.title("H2 Manufacturing LCI Data Explorer")

# Feature flag for PDF export (disabled by default)
ENABLE_PDF_EXPORT = False

# Define pathway mappings with file labels
PATHWAY_MAPPINGS = {
    "Autothermal Reforming": {
        "files": [
            {
                "label": "Without Carbon Capture",
                "source_file": "hydrogen_autothermal reforming of natural gas, without carbon capture; at plant; 99.9%, 62.2 bar"
            },
            {
                "label": "With Carbon Capture",
                "source_file": "hydrogen_autothermal_reforming_of_natural_gas_with"
            }
        ],
        "description": "This unit process produces hydrogen from deionized water, electricity and natural gas via autothermal reforming (ATR) pathway without carbon capture."
    },
    "Biomass Gasification": {
        "files": [
            {
                "label": "With Carbon Capture",
                "source_file": "hydrogen_wood_biomass_gasification_with_carbon_cap"
            },
            {
                "label": "Without Carbon Capture",
                "source_file": "hydrogen_wood_biomass_gasification_without_carbon_"
            }
        ],
        "description": "This unit process produces hydrogen from wood biomass (wood chips) via gasification pathway (with and without carbon capture)."
    },
    "PEM Electrolysis": {
        "files": [
            {
                "label": "PEM Electrolysis",
                "source_file": "hydrogen_pem_electrolysis_at_plant_99_9_62_3_bar"
            }
        ],
        "description": "This unit process produces hydrogen from deionized water and electricity via PEM electrolysis pathway."
    }
}

# Create two columns with equal width
left_col, right_col = st.columns(2)

with left_col:
    # Create dropdown for Pathway selection
    pathway_options = list(PATHWAY_MAPPINGS.keys())
    selected_pathway = st.selectbox(
        "Pathway File:",
        options=pathway_options,
        help="Select a pathway to view its data"
    )
    
    # Get the pathway info
    pathway_info = PATHWAY_MAPPINGS[selected_pathway]
    pathway_files = pathway_info["files"]
    
    # Show nested dropdown if multiple files, otherwise use the single file
    selected_source_file = None
    if len(pathway_files) > 1:
        # Show nested dropdown for file selection
        file_labels = [f["label"] for f in pathway_files]
        selected_file_label = st.selectbox(
            "Select File Variant:",
            options=file_labels,
            help="Select which variant of this pathway to view"
        )
        # Find the selected file
        selected_file_info = next(f for f in pathway_files if f["label"] == selected_file_label)
        selected_source_file = selected_file_info["source_file"]
    else:
        # Only one file, use it directly
        selected_source_file = pathway_files[0]["source_file"]
    
    # Find matching source file from the data (handle partial matches)
    available_source_files = general_info_df['Source_File'].unique()
    matching_source_files = []
    
    # Try exact match first
    exact_matches = [sf for sf in available_source_files if sf == selected_source_file]
    if exact_matches:
        matching_source_files.extend(exact_matches)
    else:
        # Try partial match (case-insensitive)
        partial_matches = [sf for sf in available_source_files 
                         if selected_source_file.lower().replace('_', ' ') in sf.lower().replace('_', ' ') 
                         or sf.lower().replace('_', ' ') in selected_source_file.lower().replace('_', ' ')]
        matching_source_files.extend(partial_matches)
    
    # Remove duplicates while preserving order
    matching_source_files = list(dict.fromkeys(matching_source_files))
    
    # Display pathway description
    st.markdown(f'<p style="color: #808080; font-size: 0.9em; margin-top: 10px;">{pathway_info["description"]}</p>', unsafe_allow_html=True)

    st.markdown("### Input Data")

    if selected_pathway and matching_source_files:
        # Filter and display inputs for all matching source files
        filtered_inputs = inputs_df[inputs_df['Source_File'].isin(matching_source_files)]
        
        if not filtered_inputs.empty:
            # Get available locations and ensure they're strings
            available_locations = unique_locations_df['Location_Locations'].astype(str).unique().tolist()
            
            # Ensure the Location column values are in the available options
            # Include Contribution Category if it exists in the source data
            base_columns = ['Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']
            if 'Contribution Category' in filtered_inputs.columns:
                base_columns.append('Contribution Category')
            display_df = filtered_inputs[base_columns].copy()
            display_df['Location'] = display_df['Location'].astype(str)
            
            # Add Contribution Category column if it doesn't exist, initialize with simple defaults
            if 'Contribution Category' not in display_df.columns:
                display_df['Contribution Category'] = ""
            
            # Set simple defaults for empty Contribution Category values
            default_options = ["Electricity", "Materials", "Water Supply"]
            for idx, row in display_df.iterrows():
                current_contrib = display_df.loc[idx, 'Contribution Category']
                if pd.isna(current_contrib) or str(current_contrib).strip() == "":
                    # Cycle through defaults or use first one
                    default_idx = idx % len(default_options)
                    display_df.loc[idx, 'Contribution Category'] = default_options[default_idx]
            
            edited_inputs = st.data_editor(
                display_df,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value", 
                        step=0.01,
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Location",
                        help="Select a location",
                        options=available_locations,
                    ),
                    "Contribution Category": st.column_config.TextColumn(
                        "Contribution Category",
                        help="Edit the contribution category. Defaults: Electricity, Materials, Water Supply. You can type any custom category.",
                        default="",
                    )
                },
                hide_index=True,
                key="inputs_editor",
                use_container_width=True
            )
            # Always use the edited inputs (data editor automatically saves changes)
            saved_df_inputs = edited_inputs.copy()
            
            # Ensure Contribution Category is properly saved
            if 'Contribution Category' in saved_df_inputs.columns:
                # Replace empty strings with NaN, then fill with Category if needed
                saved_df_inputs['Contribution Category'] = saved_df_inputs['Contribution Category'].replace('', pd.NA)
                saved_df_inputs['Contribution Category'] = saved_df_inputs['Contribution Category'].fillna(saved_df_inputs['Category'])
            
            update_inputs = st.button("Update Input Values", key="update_inputs")
            
            if update_inputs:
                # Save to CSV file
                try:
                    # Ensure Contribution Category column exists in inputs_df
                    if 'Contribution Category' not in inputs_df.columns:
                        inputs_df['Contribution Category'] = ''
                    
                    # Get the original indices of filtered rows
                    filtered_indices = filtered_inputs.index.tolist()
                    
                    # Update the original inputs_df with the edited values
                    for i, orig_idx in enumerate(filtered_indices):
                        if i < len(saved_df_inputs) and orig_idx < len(inputs_df):
                            edited_row = saved_df_inputs.iloc[i]
                            
                            # Update Contribution Category (can be custom text)
                            contrib_cat = edited_row.get('Contribution Category', '')
                            if pd.notna(contrib_cat) and str(contrib_cat).strip() and str(contrib_cat).lower() != 'nan':
                                inputs_df.loc[orig_idx, 'Contribution Category'] = str(contrib_cat).strip()
                            
                            # Update other editable fields
                            if 'Amount' in edited_row:
                                inputs_df.loc[orig_idx, 'Amount'] = edited_row['Amount']
                            if 'Location' in edited_row:
                                inputs_df.loc[orig_idx, 'Location'] = edited_row['Location']
                    
                    # Save updated inputs to CSV
                    inputs_df.to_csv(INPUTS_FILE, index=False)
                    st.success(f"✓ Input values saved to {INPUTS_FILE}")
                    st.info("💡 Your custom Contribution Category values have been saved!")
                    # Reload the data to show updates
                    inputs_df = pd.read_csv(INPUTS_FILE)
                    st.rerun()  # Refresh to show updated data
                except Exception as e:
                    st.error(f"Error saving input values: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.info("No input data available for the selected pathway.")
            saved_df_inputs = pd.DataFrame()
    else:
        saved_df_inputs = pd.DataFrame()

    st.markdown("### Output Data")
    
    if selected_pathway and matching_source_files:
        # Filter and display outputs for all matching source files
        filtered_outputs = outputs_df[outputs_df['Source_File'].isin(matching_source_files)]
        
        if not filtered_outputs.empty:
            # Get available locations and ensure they're strings
            available_locations = unique_locations_df['Location_Locations'].astype(str).unique().tolist()
            
            # Ensure the Location column values are in the available options
            # Include Contribution Category if it exists in the source data
            base_columns = ['Is reference?', 'Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']
            if 'Contribution Category' in filtered_outputs.columns:
                base_columns.append('Contribution Category')
            display_df = filtered_outputs[base_columns].copy()
            display_df['Location'] = display_df['Location'].astype(str)
            
            # Add Contribution Category column if it doesn't exist, initialize with simple defaults
            if 'Contribution Category' not in display_df.columns:
                display_df['Contribution Category'] = ""
            
            # Set simple defaults for empty Contribution Category values
            default_options = ["Electricity", "Materials", "Water Supply"]
            for idx, row in display_df.iterrows():
                current_contrib = display_df.loc[idx, 'Contribution Category']
                if pd.isna(current_contrib) or str(current_contrib).strip() == "":
                    # Cycle through defaults or use first one
                    default_idx = idx % len(default_options)
                    display_df.loc[idx, 'Contribution Category'] = default_options[default_idx]
            
            edited_outputs = st.data_editor(
                display_df,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value",
                        step=0.01,
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Location",
                        help="Select a location",
                        options=available_locations,
                    ),
                    "Contribution Category": st.column_config.TextColumn(
                        "Contribution Category",
                        help="Edit the contribution category. Defaults: Electricity, Materials, Water Supply. You can type any custom category.",
                        default="",
                    )
                },
                hide_index=True,
                key="outputs_editor",
                use_container_width=True
            )
            # Always use the edited outputs (data editor automatically saves changes)
            saved_df_outputs = edited_outputs.copy()
            
            # Ensure Contribution Category is properly saved
            if 'Contribution Category' in saved_df_outputs.columns:
                # Replace empty strings with NaN, then fill with Category if needed
                saved_df_outputs['Contribution Category'] = saved_df_outputs['Contribution Category'].replace('', pd.NA)
                saved_df_outputs['Contribution Category'] = saved_df_outputs['Contribution Category'].fillna(saved_df_outputs['Category'])
            
            update_outputs = st.button("Update Output Values", key="update_outputs")
            
            if update_outputs:
                # Save to CSV file
                try:
                    # Ensure Contribution Category column exists in outputs_df
                    if 'Contribution Category' not in outputs_df.columns:
                        outputs_df['Contribution Category'] = ''
                    
                    # Get the original indices of filtered rows
                    filtered_indices = filtered_outputs.index.tolist()
                    
                    # Update the original outputs_df with the edited values
                    for i, orig_idx in enumerate(filtered_indices):
                        if i < len(saved_df_outputs) and orig_idx < len(outputs_df):
                            edited_row = saved_df_outputs.iloc[i]
                            
                            # Update Contribution Category (can be custom text)
                            contrib_cat = edited_row.get('Contribution Category', '')
                            if pd.notna(contrib_cat) and str(contrib_cat).strip() and str(contrib_cat).lower() != 'nan':
                                outputs_df.loc[orig_idx, 'Contribution Category'] = str(contrib_cat).strip()
                            
                            # Update other editable fields
                            if 'Amount' in edited_row:
                                outputs_df.loc[orig_idx, 'Amount'] = edited_row['Amount']
                            if 'Location' in edited_row:
                                outputs_df.loc[orig_idx, 'Location'] = edited_row['Location']
                    
                    # Save updated outputs to CSV
                    outputs_df.to_csv(OUTPUTS_FILE, index=False)
                    st.success(f"✓ Output values saved to {OUTPUTS_FILE}")
                    st.info("💡 Your custom Contribution Category values have been saved!")
                    # Reload the data to show updates
                    outputs_df = pd.read_csv(OUTPUTS_FILE)
                    st.rerun()  # Refresh to show updated data
                except Exception as e:
                    st.error(f"Error saving output values: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.info("No output data available for the selected pathway.")
            saved_df_outputs = pd.DataFrame()
    else:
        saved_df_outputs = pd.DataFrame()

    # Feature flag for semantic flow mapping (disabled by default)
    ENABLE_SEMANTIC_MAPPING = False

    # Import mapping function
    from utils.mapper import map_flows

    # Initialize mapping DataFrames
    inputs_mapping_df = pd.DataFrame()
    outputs_mapping_df = pd.DataFrame()

    # Map both inputs and outputs to database
    try:
        for df_name, df in [('inputs', saved_df_inputs), ('outputs', saved_df_outputs)]:
            if not df.empty:
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
                    # Always add Contribution Category (it's set in mapper with Category as fallback)
                    record['Contribution Category'] = mapping.get('contribution_category', mapping['category'])
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

    st.markdown("### Pathway Comparison")
    
    # Select up to 3 pathways for comparison
    st.markdown("#### Select Pathways for Comparison (up to 3)")
    st.info("Select up to 3 pathways to compare their data side by side.")
    
    # Create pathway options with file variants
    def get_pathway_options():
        """Generate a list of pathway options with file variants"""
        options = []
        for pathway_name, pathway_info in PATHWAY_MAPPINGS.items():
            for file_info in pathway_info["files"]:
                option_label = f"{pathway_name} - {file_info['label']}"
                options.append({
                    "label": option_label,
                    "pathway": pathway_name,
                    "file_label": file_info["label"],
                    "source_file": file_info["source_file"]
                })
        return options
    
    pathway_options_list = get_pathway_options()
    pathway_option_labels = [opt["label"] for opt in pathway_options_list]
    
    # Three columns for selecting up to 3 pathways
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    selected_comparison_pathways = []
    
    with comp_col1:
        pathway1_comp = st.selectbox(
            "Pathway 1:",
            options=["None"] + pathway_option_labels,
            key="pathway1_comp"
        )
        if pathway1_comp != "None":
            selected_comparison_pathways.append(next(opt for opt in pathway_options_list if opt["label"] == pathway1_comp))
    
    with comp_col2:
        pathway2_comp = st.selectbox(
            "Pathway 2:",
            options=["None"] + pathway_option_labels,
            key="pathway2_comp"
        )
        if pathway2_comp != "None" and pathway2_comp not in [p["label"] for p in selected_comparison_pathways]:
            selected_comparison_pathways.append(next(opt for opt in pathway_options_list if opt["label"] == pathway2_comp))
    
    with comp_col3:
        pathway3_comp = st.selectbox(
            "Pathway 3:",
            options=["None"] + pathway_option_labels,
            key="pathway3_comp"
        )
        if pathway3_comp != "None" and pathway3_comp not in [p["label"] for p in selected_comparison_pathways]:
            selected_comparison_pathways.append(next(opt for opt in pathway_options_list if opt["label"] == pathway3_comp))
    
    # Display comparison if pathways are selected
    if selected_comparison_pathways:
        st.markdown("---")
        st.markdown("#### Pathway Comparison Results")
        
        # Import calculation function
        from utils.calculate import calculate_impacts
        from utils.mapper import map_flows
        
        # Find matching source files for each selected pathway
        available_source_files = general_info_df['Source_File'].unique()
        
        comparison_data = []
        for pathway_opt in selected_comparison_pathways:
            source_file = pathway_opt["source_file"]
            # Find matching source file
            matching_files = []
            exact_matches = [sf for sf in available_source_files if sf == source_file]
            if exact_matches:
                matching_files.extend(exact_matches)
            else:
                partial_matches = [sf for sf in available_source_files 
                                 if source_file.lower().replace('_', ' ') in sf.lower().replace('_', ' ') 
                                 or sf.lower().replace('_', ' ') in source_file.lower().replace('_', ' ')]
                matching_files.extend(partial_matches)
            
            if matching_files:
                # Get inputs and outputs for this pathway
                pathway_inputs = inputs_df[inputs_df['Source_File'].isin(matching_files)]
                pathway_outputs = outputs_df[outputs_df['Source_File'].isin(matching_files)]
                
                # Count flows
                num_input_flows = len(pathway_inputs)
                num_output_flows = len(pathway_outputs)
                total_flows = num_input_flows + num_output_flows
                
                # Calculate CO2 eq impacts
                total_co2_eq = 0.0
                try:
                    # Map inputs and outputs
                    if not pathway_inputs.empty:
                        inputs_mapping = map_flows(pathway_inputs, MAPPING_FILE)
                        inputs_mapping_df = pd.DataFrame([
                            {
                                'Mapped Flow': mapping['mapped_flow'],
                                'Amount': mapping['amount'],
                                'Unit': mapping['unit'],
                                'Category': mapping['category'],
                                'Contribution Category': mapping.get('contribution_category', mapping['category'])
                            }
                            for mapping in inputs_mapping.values()
                        ])
                        
                        if not inputs_mapping_df.empty:
                            inputs_results = calculate_impacts(inputs_mapping_df, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)")
                            if not inputs_results.empty and 'Calculated Result' in inputs_results.columns:
                                total_co2_eq += inputs_results['Calculated Result'].sum()
                    
                    if not pathway_outputs.empty:
                        outputs_mapping = map_flows(pathway_outputs, MAPPING_FILE)
                        outputs_mapping_df = pd.DataFrame([
                            {
                                'Mapped Flow': mapping['mapped_flow'],
                                'Amount': mapping['amount'],
                                'Unit': mapping['unit'],
                                'Category': mapping['category'],
                                'Contribution Category': mapping.get('contribution_category', mapping['category'])
                            }
                            for mapping in outputs_mapping.values()
                        ])
                        
                        if not outputs_mapping_df.empty:
                            outputs_results = calculate_impacts(outputs_mapping_df, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)")
                            if not outputs_results.empty and 'Calculated Result' in outputs_results.columns:
                                total_co2_eq += outputs_results['Calculated Result'].sum()
                except Exception as e:
                    st.warning(f"Could not calculate CO2 eq for {pathway_opt['label']}: {str(e)}")
                
                # Use CO2 eq as GWP (they're equivalent for carbon footprint)
                total_gwp = total_co2_eq
                
                comparison_data.append({
                    "Pathway": pathway_opt["label"],
                    "Number of Flows": total_flows,
                    "CO₂ eq (kg CO₂ eq)": total_co2_eq,
                    "Total GWP (kg CO₂ eq)": total_gwp
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Calculate average GWP
            avg_gwp = comparison_df['Total GWP (kg CO₂ eq)'].mean() if not comparison_df.empty else 0
            
            # Display summary metrics
            st.markdown("##### Summary Metrics")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Number of Pathways", len(comparison_df))
            
            with summary_col2:
                total_flows_all = comparison_df['Number of Flows'].sum()
                st.metric("Total Flows", total_flows_all)
            
            with summary_col3:
                st.metric("Average GWP", f"{avg_gwp:.3f} kg CO₂ eq")
            
            with summary_col4:
                total_gwp_all = comparison_df['Total GWP (kg CO₂ eq)'].sum()
                st.metric("Total GWP (All Pathways)", f"{total_gwp_all:.3f} kg CO₂ eq")
            
            st.markdown("---")
            st.markdown("##### Detailed Comparison")
            
            # Display comparison table
            st.dataframe(
                comparison_df,
                column_config={
                    "Pathway": st.column_config.Column("Pathway", width="large"),
                    "Number of Flows": st.column_config.NumberColumn("Number of Flows", width="medium", format="%d"),
                    "CO₂ eq (kg CO₂ eq)": st.column_config.NumberColumn(
                        "CO₂ eq (kg CO₂ eq)", 
                        width="medium",
                        format="%.3f"
                    ),
                    "Total GWP (kg CO₂ eq)": st.column_config.NumberColumn(
                        "Total GWP (kg CO₂ eq)", 
                        width="medium",
                        format="%.3f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Store selected pathways in session state for potential future use
            st.session_state['selected_comparison_pathways'] = selected_comparison_pathways
            st.session_state['pathway_comparison_results'] = comparison_df

with right_col:
    # Import calculation function 
    from utils.calculate import calculate_impacts
    from utils.visualize import generate_impact_piechart, generate_impact_barchart

    st.markdown("### GWP Analysis")
    
    # Add GWP analysis section
    if st.button("Calculate GWP for All Pathways"):
        try:
            from process_lci import LCIProcessor
            
            with st.spinner("Processing all LCI data..."):
                processor = LCIProcessor()
                lci_data = processor.load_all_lci_data()
                recipe_data = processor.load_recipe_data()
                gwp_results = processor.calculate_gwp()
            
            if not gwp_results.empty:
                st.markdown("#### Pathway GWP Results")
                st.dataframe(
                    gwp_results,
                    column_config={
                        "total_gwp_kgco2e": st.column_config.NumberColumn(
                            "Total GWP (kg CO₂-eq)", 
                            help="Total Global Warming Potential",
                            format="%.3f"
                        ),
                        "num_flows": st.column_config.NumberColumn(
                            "Number of Flows",
                            help="Total flows in this pathway"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_gwp = gwp_results.iloc[-1]['total_gwp_kgco2e']
                    st.metric("Best Pathway GWP", f"{best_gwp:.3f} kg CO₂-eq")
                
                with col2:
                    worst_gwp = gwp_results.iloc[0]['total_gwp_kgco2e']
                    st.metric("Worst Pathway GWP", f"{worst_gwp:.3f} kg CO₂-eq")
                
                with col3:
                    if worst_gwp > 0:
                        improvement = ((worst_gwp - best_gwp) / worst_gwp * 100)
                        st.metric("Improvement Potential", f"{improvement:.1f}%")
                
                # Download results
                csv = gwp_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download GWP Results",
                    data=csv,
                    file_name="pathway_gwp_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No GWP results available")
                
        except Exception as e:
            st.error(f"Error calculating GWP: {str(e)}")
            st.info("Make sure the LCI Excel files are in the 'input/exported LCI models/' directory")

    st.markdown("### Impact Results")

    # Initialize results DataFrames
    inputs_results_df = pd.DataFrame()
    outputs_results_df = pd.DataFrame()
    total_results_df = pd.DataFrame()

    # Check for semantic mapping results to use for impact calculation
    if 'semantic_mapped_flows' in st.session_state:
        semantic_mapped = st.session_state['semantic_mapped_flows']
        if not semantic_mapped.empty:
            st.info("📊 Using semantic mapping results for impact calculation")
            # Use semantic mapped flows as inputs
            semantic_inputs_df = semantic_mapped.copy()
            semantic_inputs_df = semantic_inputs_df.rename(columns={'Mapped Flow': 'Mapped Flow'})
            
            column_of_interest = "Carbon footprint (kg CO2 equiv.)"
            try:
                semantic_results_df = calculate_impacts(semantic_inputs_df, IDEMAT_SHEET, column_of_interest)
                if not semantic_results_df.empty:
                    semantic_results_df['Type'] = 'Input (Semantic Mapping)'
                    # Add to total results
                    if total_results_df.empty:
                        total_results_df = semantic_results_df.copy()
                    else:
                        total_results_df = pd.concat([total_results_df, semantic_results_df], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not calculate impacts for semantic mapping: {str(e)}")
    
    # Calculate inputs impacts
    if not inputs_mapping_df.empty:
        column_of_interest = "Carbon footprint (kg CO2 equiv.)"
        
        try:
            inputs_results_df = calculate_impacts(inputs_mapping_df, IDEMAT_SHEET, column_of_interest)
        except Exception as e:
            st.error(f"Error calculating inputs impacts: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            inputs_results_df = pd.DataFrame()
        
        if not inputs_results_df.empty:
            # Add a Type column to identify input vs output
            inputs_results_df['Type'] = 'Input'

    # Calculate outputs impacts
    if not outputs_mapping_df.empty:
        column_of_interest = "Carbon footprint (kg CO2 equiv.)"
        
        try:
            outputs_results_df = calculate_impacts(outputs_mapping_df, IDEMAT_SHEET, column_of_interest)
        except Exception as e:
            st.error(f"Error calculating outputs impacts: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            outputs_results_df = pd.DataFrame()
        
        if not outputs_results_df.empty:
            # Add a Type column to identify input vs output
            outputs_results_df['Type'] = 'Output'

    # Combine inputs and outputs into total impact
    if not inputs_results_df.empty and not outputs_results_df.empty:
        total_results_df = pd.concat([inputs_results_df, outputs_results_df], ignore_index=True)
    elif not inputs_results_df.empty:
        total_results_df = inputs_results_df.copy()
    elif not outputs_results_df.empty:
        total_results_df = outputs_results_df.copy()
    
    # Display combined total impact table
    if not total_results_df.empty:
        # Store in session state for PDF export
        st.session_state['impact_results'] = total_results_df
        st.session_state['selected_pathway'] = selected_pathway if 'selected_pathway' in locals() else None
        if 'matching_source_files' in locals() and matching_source_files:
            st.session_state['filtered_inputs_for_pdf'] = inputs_df[inputs_df['Source_File'].isin(matching_source_files)]
            st.session_state['filtered_outputs_for_pdf'] = outputs_df[outputs_df['Source_File'].isin(matching_source_files)]
        
        st.markdown("#### Total Impact")
        st.dataframe(
            total_results_df,
            column_config={
                "Type": st.column_config.Column("Type", help="Input or Output"),
                "Mapped Flow": st.column_config.Column("Mapped Flow", help="Flow in database"),
                "Calculated Result": st.column_config.NumberColumn(
                    "CO₂ eq Footprint (kg CO₂ eq)", 
                    help="Calculated impact",
                    format="%.3f"
                ),
                "Contribution Category": st.column_config.Column("Contribution Category", help="Category for visualization")
            },
            hide_index=True,
            use_container_width=True
        )
        
        total_impact = total_results_df['Calculated Result'].sum()
        st.metric(
            label="Total CO₂ eq Impact", 
            value=f"{total_impact:.3f} kg CO₂ eq"
        )
        
    else:
        if inputs_mapping_df.empty and outputs_mapping_df.empty:
            st.info("No input or output mapping data available")
        else:
            st.warning("No impact results calculated")

    st.markdown("### Visualization")
    
    
    chart_type = st.selectbox(
        "Select chart type",
        ["Pie Chart", "Bar Chart", "Line Chart"]
    )

    if st.button("Generate Chart"):
        # Use total_results_df if available, otherwise fallback to inputs_results_df or outputs_results_df
        chart_data = total_results_df if not total_results_df.empty else (inputs_results_df if not inputs_results_df.empty else outputs_results_df)
        
        if chart_data.empty:
            st.warning("No data available for chart generation")
        else:
            if chart_type == "Pie Chart":
                pie_chart = generate_impact_piechart(chart_data)
                st.pyplot(pie_chart)
            elif chart_type == "Bar Chart":
                bar_chart = generate_impact_barchart(chart_data)
                st.pyplot(bar_chart)
            elif chart_type == "Line Chart": 
                st.info("Line chart coming soon!")
