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

    # # Header with title and PDF export button
    # col_title, col_export = st.columns([4, 1])

    # with col_title:
    #     st.title("H2 Manufacturing LCI Data Explorer")

    # with col_export:
    #     st.markdown("<br>", unsafe_allow_html=True)  # spacing
    #     # PDF Export button - COMMENTED OUT
    #     if False:  # st.button("📄 Export PDF", type="secondary", help="Export the full page as a PDF report", key="pdf_export_btn_top"):
    #         if not REPORTLAB_AVAILABLE:
    #             st.error("❌ PDF export requires 'reportlab' library.")
    #             st.info("💡 Install with: `pip install reportlab`")
    #             st.code("pip install reportlab", language="bash")
    #             st.stop()
            
    #         try:
    #             from utils.pdf_export import create_pdf_report
    #             from datetime import datetime
                
    #             # Get data from session state or current selections
    #             selected_pathway_for_pdf = st.session_state.get('selected_pathway', None)
    #             filtered_inputs_for_pdf = st.session_state.get('filtered_inputs_for_pdf', pd.DataFrame())
    #             filtered_outputs_for_pdf = st.session_state.get('filtered_outputs_for_pdf', pd.DataFrame())
    #             total_results_df_for_pdf = st.session_state.get('impact_results', pd.DataFrame())
                
    #             # Collect all data for PDF export
    #             app_data = {
    #                 'selected_source': selected_pathway_for_pdf,
    #                 'inputs_df': filtered_inputs_for_pdf,
    #                 'outputs_df': filtered_outputs_for_pdf,
    #                 'impact_results': total_results_df_for_pdf,
    #                 'semantic_mapping_results': st.session_state.get('semantic_mapping_results', None),
    #                 'pathway_comparison_results': st.session_state.get('pathway_comparison_results', None),
    #             }
                
    #             # Generate PDF filename with timestamp
    #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #             pdf_filename = f"LCA_Report_{timestamp}.pdf"
    #             pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
                
    #             # Ensure output directory exists
    #             os.makedirs(OUTPUT_DIR, exist_ok=True)
                
    #             with st.spinner("Generating PDF report..."):
    #                 create_pdf_report(pdf_path, app_data)
                
    #             st.success(f"✓ PDF report generated successfully!")
    #             st.info(f"📄 Report saved to: {pdf_path}")
                
    #             # Provide download button
    #             with open(pdf_path, "rb") as f:
    #                 pdf_bytes = f.read()
    #                 st.download_button(
    #                     label="📥 Download PDF Report",
    #                     data=pdf_bytes,
    #                     file_name=pdf_filename,
    #                     mime="application/pdf",
    #                     key="download_pdf_btn_top"
    #                 )
                    
    #         except ImportError as e:
    #             error_msg = str(e)
    #             if 'reportlab' in error_msg.lower() or 'reportlab' in str(type(e)).lower():
    #                 st.error("❌ PDF export requires 'reportlab' library.")
    #                 st.info("💡 Install with: `pip install reportlab`")
    #                 st.code("pip install reportlab", language="bash")
    #             else:
    #                 st.error(f"Import error: {error_msg}")
    #                 import traceback
    #                 with st.expander("Error Details"):
    #                     st.code(traceback.format_exc())
    #         except Exception as e:
    #             st.error(f"Error generating PDF: {str(e)}")
    #             import traceback
    #             with st.expander("Error Details"):
    #                 st.code(traceback.format_exc())

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

    # # Semantic Flow Mapping (Ollama) - COMMENTED OUT
    # if False:
    #     st.markdown("### Semantic Flow Mapping (Ollama)")
        
    #     # Add semantic mapping section
    #     st.markdown("#### Upload Unique Flows for Semantic Mapping")
    #     st.info("Upload a CSV file with unique flow locations. The system will use Ollama to find the most similar processes from the IDEMAT database.")
        
    #     uploaded_file = st.file_uploader(
    #     "Choose a CSV file with unique flows",
    #     type=['csv'],
    #         help="CSV should contain columns with flow names (e.g., 'Inputs_Flows', 'Outputs_Flows', 'Providers_Providers') and 'Location_Locations'"
    #     )
        
    #     # Check Ollama connection status
    #     try:
    #         import requests
    #         from config.config import OLLAMA_HOST
    #         ollama_check = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
    #         if ollama_check.status_code == 200:
    #             st.success("✓ Ollama is running and ready")
    #             ollama_available = True
    #         else:
    #             st.warning("⚠ Ollama returned an error. Please check if Ollama is running.")
    #             ollama_available = False
    #     except:
    #         st.error("✗ Cannot connect to Ollama. Please make sure Ollama is running (ollama serve)")
    #         ollama_available = False
        
    #     if uploaded_file is not None and ollama_available:
    #         # Read uploaded file
    #         try:
    #             uploaded_df = pd.read_csv(uploaded_file)
    #             st.success(f"✓ File uploaded successfully ({len(uploaded_df)} rows)")
                
    #             # Show preview
    #             with st.expander("Preview uploaded data"):
    #                 st.dataframe(uploaded_df.head(10), use_container_width=True)
                
    #             # Run semantic mapping
    #             if st.button("🚀 Run Semantic Mapping with Ollama", type="primary"):
    #                 try:
    #                     # Import semantic mapper
    #                     sys.path.append(str(Path(__file__).parent / "scripts"))
    #                     from scripts.semantic_mapping import SemanticMapper
                        
    #                     with st.spinner("Initializing semantic mapper..."):
    #                         mapper = SemanticMapper()
                        
    #                     # Load processes
    #                     with st.spinner("Loading processes from IDEMAT datasheet..."):
    #                         processes = mapper.load_processes(IDEMAT_SHEET)
    #                         st.info(f"Loaded {len(processes)} processes from IDEMAT")
                        
    #                     # Extract unique flows from uploaded file
    #                     with st.spinner("Extracting unique flows from uploaded file..."):
    #                         flows_df = mapper.load_unique_flows_from_dataframe(uploaded_df)
    #                         st.info(f"Extracted {len(flows_df)} unique flow-location pairs")
                        
    #                     if len(flows_df) > 0 and len(processes) > 0:
    #                         # Load or compute process embeddings (uses cache)
    #                         if len(flows_df) > 0 and len(processes) > 0:
    #                             # Load or compute process embeddings (uses cache)
    #                             st.info("🔄 Loading process embeddings (using cache if available)...")
    #                             progress_bar = st.progress(0)
    #                             status_text = st.empty()
                                
    #                             # Check cache status first
    #                             cache_path = mapper.get_cache_path(IDEMAT_SHEET)
    #                             cache_exists = cache_path.exists()
                                
    #                             if cache_exists:
    #                                 try:
    #                                     import os
    #                                     cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    #                                     st.info(f"💾 Cache file found: {cache_size_mb:.2f} MB - Will load from disk (persistent across app restarts)")
    #                                 except:
    #                                     pass
                                
    #                             # Check cache first
    #                             cached_embeddings = mapper.load_cached_embeddings(IDEMAT_SHEET)
                                
    #                             if cached_embeddings is not None:
    #                                 # Verify all processes are in cache
    #                                 missing_processes = [p for p in processes if p not in cached_embeddings]
    #                                 if not missing_processes:
    #                                     st.success(f"✓ Loaded {len(cached_embeddings)} process embeddings from cache (instant!)")
    #                                     st.info("💡 This cache persists across app restarts - no need to recompute!")
    #                                     process_embeddings = cached_embeddings
    #                                 else:
    #                                     st.info(f"Cache missing {len(missing_processes)} processes. Computing missing embeddings...")
    #                                     process_embeddings = cached_embeddings.copy()
    #                                     # Compute missing embeddings
    #                                     for i, process in enumerate(missing_processes):
    #                                         progress = (i / len(missing_processes)) * 0.5
    #                                         progress_bar.progress(progress)
    #                                         status_text.text(f"Computing missing embedding {i+1}/{len(missing_processes)}...")
    #                                         embedding = mapper.get_embedding(process)
    #                                         if embedding:
    #                                             process_embeddings[process] = embedding
    #                                         time.sleep(0.1)
    #                                     # Save updated cache
    #                                     mapper.save_embeddings_cache(process_embeddings, IDEMAT_SHEET)
    #                                     st.success(f"✓ Updated cache with {len(process_embeddings)} process embeddings")
    #                             else:
    #                                 # No cache, compute all embeddings
    #                                 st.info("⏳ No cache found. Computing embeddings for all processes (this will take several minutes, but will be cached for future use)...")
    #                                 process_embeddings = {}
    #                                 total_processes = len(processes)
                                    
    #                                 for i, process in enumerate(processes):
    #                                     if i % 50 == 0:
    #                                         progress = (i / total_processes) * 0.5  # First 50% for processes
    #                                         progress_bar.progress(progress)
    #                                         status_text.text(f"Processing process {i+1}/{total_processes}... ({100*(i+1)/total_processes:.1f}%)")
                                        
    #                                     embedding = mapper.get_embedding(process)
    #                                     if embedding:
    #                                         process_embeddings[process] = embedding
    #                                     time.sleep(0.1)
                                    
    #                                 # Save to cache
    #                                 mapper.save_embeddings_cache(process_embeddings, IDEMAT_SHEET)
    #                                 st.success(f"✓ Computed and cached embeddings for {len(process_embeddings)} processes")
    #                                 st.info("💾 Cache saved to disk - will persist across app restarts and reloads!")
                                
    #                             # Map flows
    #                             st.info("⏳ Finding most similar processes for each flow...")
    #                             mapping_results = []
    #                             total_flows = len(flows_df)
                                
    #                             for idx, row in flows_df.iterrows():
    #                                 flow_name = row['Unique Flow Name']
    #                                 location = row.get('Location', '')
                                    
    #                                 progress = 0.5 + ((idx + 1) / total_flows) * 0.5  # Second 50% for flows
    #                                 progress_bar.progress(progress)
    #                                 status_text.text(f"Mapping flow {idx+1}/{total_flows}: {flow_name}")
                                    
    #                                 similar_process, similarity_score = mapper.find_most_similar_process(
    #                                     flow_name,
    #                                     process_embeddings
    #                                 )
                                    
    #                                 mapping_results.append({
    #                                     'Unique Flow Name': flow_name,
    #                                     'Location': location if location else "",
    #                                     'Most Similar Process': similar_process,
    #                                     'Similarity Score': similarity_score
    #                                 })
                                
    #                             progress_bar.progress(1.0)
    #                             status_text.text("✓ Mapping completed!")
                                
    #                             # Create results DataFrame
    #                             mapping_df = pd.DataFrame(mapping_results)
                                
    #                             # Store in session state
    #                             st.session_state['semantic_mapping_results'] = mapping_df
                                
    #                             st.success(f"✓ Successfully mapped {len(mapping_df)} flows!")
                                
    #                             # Display results
    #                             st.markdown("#### Mapping Results")
    #                             display_df = mapping_df[['Unique Flow Name', 'Location', 'Most Similar Process']].copy()
    #                             st.dataframe(
    #                                 display_df,
    #                                 column_config={
    #                                     "Unique Flow Name": st.column_config.Column("Flow Name", width="large"),
    #                                     "Location": st.column_config.Column("Location", width="medium"),
    #                                     "Most Similar Process": st.column_config.Column("Most Similar Process", width="large")
    #                                 },
    #                                 hide_index=True,
    #                                 use_container_width=True
    #                             )
                                
    #                             # Download button
    #                             csv = mapping_df[['Unique Flow Name', 'Location', 'Most Similar Process']].to_csv(index=False)
    #                             st.download_button(
    #                                 label="📥 Download Mapping Results (CSV)",
    #                                 data=csv,
    #                                 file_name="semantic_similarity_table.csv",
    #                                 mime="text/csv"
    #                             )
                                
    #                             # Show similarity scores
    #                             with st.expander("View Similarity Scores"):
    #                                 st.dataframe(
    #                                     mapping_df[['Unique Flow Name', 'Most Similar Process', 'Similarity Score']].sort_values('Similarity Score', ascending=False),
    #                                     use_container_width=True
    #                                 )
                                
    #                             # Option to use semantic mapping for impact calculations
    #                             st.markdown("#### Use Semantic Mapping for Impact Calculations")
    #                             st.info("To calculate impacts, you need to provide flow data with amounts. You can either:")
    #                             st.markdown("1. **Use existing flow data**: Select a source file above and the semantic mapping will be used automatically")
    #                             st.markdown("2. **Upload flow data with amounts**: Upload a CSV with flows, amounts, units, and categories")
                                
    #                             # Option to upload flow data with amounts
    #                             flow_data_file = st.file_uploader(
    #                                 "Upload flow data with amounts (optional)",
    #                                 type=['csv'],
    #                                 help="CSV should contain: Flow, Amount, Unit, Category, and optionally Location"
    #                             )
                                
    #                             if flow_data_file is not None:
    #                                 try:
    #                                     flow_data_df = pd.read_csv(flow_data_file)
    #                                     st.success(f"✓ Flow data uploaded ({len(flow_data_df)} rows)")
                                        
    #                                     # Check required columns
    #                                     required_cols = ['Flow', 'Amount', 'Unit', 'Category']
    #                                     missing_cols = [col for col in required_cols if col not in flow_data_df.columns]
                                        
    #                                     if missing_cols:
    #                                         st.error(f"Missing required columns: {', '.join(missing_cols)}")
    #                                     else:
    #                                         # Merge with semantic mapping
    #                                         if st.button("🚀 Calculate Impacts with Semantic Mapping", type="primary"):
    #                                             # Create mapping dictionary from semantic results
    #                                             semantic_mapping_dict = {}
    #                                             for _, row in mapping_df.iterrows():
    #                                                 flow_name = row['Unique Flow Name']
    #                                                 location = row.get('Location', '')
    #                                                 mapped_process = row['Most Similar Process']
                                                    
    #                                                 # Create composite key
    #                                                 if location:
    #                                                     key = f"{flow_name}|{location}"
    #                                                 else:
    #                                                     key = flow_name
    #                                                 semantic_mapping_dict[key] = mapped_process
                                                
    #                                             # Apply mapping to flow data
    #                                             mapped_flows = []
    #                                             for _, flow_row in flow_data_df.iterrows():
    #                                                 flow_name = flow_row['Flow']
    #                                                 location = flow_row.get('Location', '')
                                                    
    #                                                 # Try to find mapping
    #                                                 mapped_process = None
    #                                                 if location:
    #                                                     key = f"{flow_name}|{location}"
    #                                                     mapped_process = semantic_mapping_dict.get(key)
                                                    
    #                                                 if not mapped_process:
    #                                                     mapped_process = semantic_mapping_dict.get(flow_name, flow_name)
                                                    
    #                                                 mapped_flows.append({
    #                                                     'Mapped Flow': mapped_process,
    #                                                     'Amount': flow_row['Amount'],
    #                                                     'Unit': flow_row['Unit'],
    #                                                     'Category': flow_row['Category'],
    #                                                     'Contribution Category': flow_row.get('Contribution Category', flow_row['Category']),
    #                                                     'Original Flow': flow_name
    #                                                 })
                                                
    #                                             # Store in session state for impact calculation
    #                                             st.session_state['semantic_mapped_flows'] = pd.DataFrame(mapped_flows)
    #                                             st.success(f"✓ Mapped {len(mapped_flows)} flows using semantic mapping!")
    #                                             st.info("Scroll down to 'Impact Results' section to see calculated impacts")
                                                
    #                                 except Exception as e:
    #                                     st.error(f"Error processing flow data: {str(e)}")
    #                                     import traceback
    #                                     with st.expander("Error Details"):
    #                                         st.code(traceback.format_exc())
    #                         else:
    #                             st.error("No flows or processes loaded. Cannot perform mapping.")
                        
    #             except Exception as e:
    #                 st.error(f"Error during semantic mapping: {str(e)}")
    #                 import traceback
    #                 with st.expander("Error Details"):
    #                     st.code(traceback.format_exc())
    #         except Exception as e:
    #             st.error(f"Error reading uploaded file: {str(e)}")
        
    #     # Show existing mapping results if available
    #     if 'semantic_mapping_results' in st.session_state:
    #         st.markdown("#### Previous Mapping Results")
    #         prev_results = st.session_state['semantic_mapping_results']
    #         st.dataframe(
    #             prev_results[['Unique Flow Name', 'Location', 'Most Similar Process']],
    #             use_container_width=True
    #         )
    
    # st.markdown("---")
    # st.markdown("### Flow Mapping")

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
