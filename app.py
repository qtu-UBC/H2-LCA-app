import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path

# Try to import reportlab at module level
try:
    import reportlab  # noqa: F401
    from reportlab.lib import colors  # noqa: F401
    from reportlab.lib.pagesizes import letter  # noqa: F401

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

# -----------------------------
# Load CSVs safely
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV: {path}\n\n{e}")
        return pd.DataFrame()

general_info_df = safe_read_csv(GENERAL_INFO_FILE)
inputs_df = safe_read_csv(INPUTS_FILE)
outputs_df = safe_read_csv(OUTPUTS_FILE)
unique_locations_df = safe_read_csv(UNIQUE_FLOWS_PROVIDERS_FILE)

# Set up the Streamlit page with full width
st.set_page_config(layout="wide")

st.title("H2 Manufacturing LCI Data Explorer")

# Feature flag for PDF export (disabled by default)
ENABLE_PDF_EXPORT = False

# Define pathway mappings with file labels
PATHWAY_MAPPINGS = {
    "Autothermal Reforming": {
        "files": [
            {
                "label": "Without Carbon Capture",
                "source_file": "hydrogen_autothermal reforming of natural gas, without carbon capture; at plant; 99.9%, 62.2 bar",
            },
            {
                "label": "With Carbon Capture",
                "source_file": "hydrogen_autothermal_reforming_of_natural_gas_with",
            },
        ],
        "description": "This unit process produces hydrogen from deionized water, electricity and natural gas via autothermal reforming (ATR) pathway without carbon capture.",
    },
    "Biomass Gasification": {
        "files": [
            {
                "label": "With Carbon Capture",
                "source_file": "hydrogen_wood_biomass_gasification_with_carbon_cap",
            },
            {
                "label": "Without Carbon Capture",
                "source_file": "hydrogen_wood_biomass_gasification_without_carbon_",
            },
        ],
        "description": "This unit process produces hydrogen from wood biomass (wood chips) via gasification pathway (with and without carbon capture).",
    },
    "PEM Electrolysis": {
        "files": [
            {
                "label": "PEM Electrolysis",
                "source_file": "hydrogen_pem_electrolysis_at_plant_99_9_62_3_bar",
            }
        ],
        "description": "This unit process produces hydrogen from deionized water and electricity via PEM electrolysis pathway.",
    },
}

left_col, right_col = st.columns(2)

# Initialize these so they always exist (prevents NameError)
saved_df_inputs = pd.DataFrame()
saved_df_outputs = pd.DataFrame()
inputs_mapping_df = pd.DataFrame()
outputs_mapping_df = pd.DataFrame()

# -----------------------------
# LEFT COLUMN
# -----------------------------
with left_col:
    pathway_options = list(PATHWAY_MAPPINGS.keys())
    selected_pathway = st.selectbox(
        "Pathway File:",
        options=pathway_options,
        help="Select a pathway to view its data",
    )

    pathway_info = PATHWAY_MAPPINGS[selected_pathway]
    pathway_files = pathway_info["files"]

    selected_source_file = None
    if len(pathway_files) > 1:
        file_labels = [f["label"] for f in pathway_files]
        selected_file_label = st.selectbox(
            "Select File Variant:",
            options=file_labels,
            help="Select which variant of this pathway to view",
        )
        selected_file_info = next(
            f for f in pathway_files if f["label"] == selected_file_label
        )
        selected_source_file = selected_file_info["source_file"]
    else:
        selected_source_file = pathway_files[0]["source_file"]

    available_source_files = (
        general_info_df["Source_File"].unique()
        if "Source_File" in general_info_df.columns
        else []
    )

    matching_source_files = []
    exact_matches = [sf for sf in available_source_files if sf == selected_source_file]
    if exact_matches:
        matching_source_files.extend(exact_matches)
    else:
        # Try partial match (case-insensitive)
        partial_matches = [
            sf
            for sf in available_source_files
            if selected_source_file.lower().replace("_", " ")
            in sf.lower().replace("_", " ")
            or sf.lower().replace("_", " ")
            in selected_source_file.lower().replace("_", " ")
        ]
        matching_source_files.extend(partial_matches)

    matching_source_files = list(dict.fromkeys(matching_source_files))

    st.markdown(
        f'<p style="color: #808080; font-size: 0.9em; margin-top: 10px;">{pathway_info["description"]}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if "expanded_section" not in st.session_state:
        st.session_state["expanded_section"] = "input_data"

    # Add custom CSS to make active buttons green
    st.markdown("""
        <style>
        /* Style active navigation buttons to be green */
        button[kind="primary"] {
            background-color: #28a745 !important;
            border-color: #28a745 !important;
            color: white !important;
        }
        button[kind="primary"]:hover {
            background-color: #218838 !important;
            border-color: #1e7e34 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

    def create_nav_button(label: str, section_key: str, button_key: str):
        current_section = st.session_state.get("expanded_section", None)
        is_active = current_section == section_key
        button_type = "primary" if is_active else "secondary"

        if st.button(label, use_container_width=True, key=button_key, type=button_type):
            # Toggle: if already expanded, collapse it; otherwise expand it
            if is_active:
                st.session_state["expanded_section"] = None
            else:
                st.session_state["expanded_section"] = section_key
            st.rerun()

    with nav_col1:
        create_nav_button("📥 Input Data", "input_data", "nav_input_data")
        create_nav_button("📤 Output Data", "output_data", "nav_output_data")

    with nav_col2:
        create_nav_button("🔗 Input Mapping", "input_mapping", "nav_input_mapping")
        create_nav_button("🔗 Output Mapping", "output_mapping", "nav_output_mapping")

    with nav_col3:
        create_nav_button("📊 Pathway Comparison", "pathway_comparison", "nav_pathway_comp")
        create_nav_button("🌍 GWP Analysis", "gwp_analysis", "nav_gwp")

    with nav_col4:
        create_nav_button("📈 Impact Results", "impact_results", "nav_impact")
        create_nav_button("📊 Visualization", "visualization", "nav_viz")

    st.markdown("---")

    # -----------------------------
    # INPUT DATA
    # -----------------------------
    with st.expander(
        "📥 **Input Data**",
        expanded=(st.session_state.get("expanded_section") == "input_data"),
    ):
        st.markdown("#### Input Data")

        # ✅ Define filtered_inputs BEFORE using it
        if selected_pathway and matching_source_files and "Source_File" in inputs_df.columns:
            filtered_inputs = inputs_df[inputs_df["Source_File"].isin(matching_source_files)]
        else:
            filtered_inputs = pd.DataFrame()
        
        if not filtered_inputs.empty:
            available_locations = (
                unique_locations_df["Location_Locations"].astype(str).unique().tolist()
                if "Location_Locations" in unique_locations_df.columns
                else []
            )

            base_columns = ["Flow", "Category", "Amount", "Unit", "Provider", "Location"]
            if "Contribution Category" in filtered_inputs.columns:
                base_columns.append("Contribution Category")

            base_columns = [c for c in base_columns if c in filtered_inputs.columns]
            display_df = filtered_inputs[base_columns].copy()

            if "Location" in display_df.columns:
                display_df["Location"] = display_df["Location"].astype(str)

            if "Contribution Category" not in display_df.columns:
                display_df["Contribution Category"] = ""

            default_options = ["Electricity", "Materials", "Water Supply"]
            if "Contribution Category" in display_df.columns:
                for idx in display_df.index:
                    current_contrib = display_df.loc[idx, "Contribution Category"]
                    if pd.isna(current_contrib) or str(current_contrib).strip() == "":
                        default_idx = (list(display_df.index).index(idx)) % len(default_options)
                        display_df.loc[idx, "Contribution Category"] = default_options[default_idx]
            
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
                    ),
                },
                hide_index=True,
                key="inputs_editor",
                use_container_width=True,
            )

            saved_df_inputs = edited_inputs.copy()
            
            if "Contribution Category" in saved_df_inputs.columns and "Category" in saved_df_inputs.columns:
                saved_df_inputs["Contribution Category"] = saved_df_inputs["Contribution Category"].replace("", pd.NA)
                saved_df_inputs["Contribution Category"] = saved_df_inputs["Contribution Category"].fillna(
                    saved_df_inputs["Category"]
                )
            
            # Store in session_state for mapping section
            st.session_state["saved_df_inputs"] = saved_df_inputs
            
            update_inputs = st.button("Update Input Values", key="update_inputs")
            
            if update_inputs:
                try:
                    if "Contribution Category" not in inputs_df.columns:
                        inputs_df["Contribution Category"] = ""
                    
                    filtered_indices = filtered_inputs.index.tolist()
                    
                    for i, orig_idx in enumerate(filtered_indices):
                        if i >= len(saved_df_inputs):
                            break
                        edited_row = saved_df_inputs.iloc[i]
                        
                        contrib_cat = edited_row.get("Contribution Category", "")
                        if pd.notna(contrib_cat) and str(contrib_cat).strip() and str(contrib_cat).lower() != "nan":
                            inputs_df.loc[orig_idx, "Contribution Category"] = str(contrib_cat).strip()

                        if "Amount" in edited_row:
                            inputs_df.loc[orig_idx, "Amount"] = edited_row["Amount"]
                        if "Location" in edited_row:
                            inputs_df.loc[orig_idx, "Location"] = edited_row["Location"]

                    inputs_df.to_csv(INPUTS_FILE, index=False)
                    st.success(f"✓ Input values saved to {INPUTS_FILE}")
                    st.info("💡 Your custom Contribution Category values have been saved!")
                    inputs_df = pd.read_csv(INPUTS_FILE)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving input values: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No input data available for the selected pathway.")
            # Use session_state if available, otherwise empty DataFrame
            saved_df_inputs = st.session_state.get("saved_df_inputs", pd.DataFrame())
    
    # Ensure saved_df_inputs is available (use session_state if not set locally)
    if "saved_df_inputs" not in locals() or saved_df_inputs.empty:
        saved_df_inputs = st.session_state.get("saved_df_inputs", pd.DataFrame())

    # -----------------------------
    # OUTPUT DATA
    # -----------------------------
    with st.expander(
        "📤 **Output Data**",
        expanded=(st.session_state.get("expanded_section") == "output_data"),
    ):
        st.markdown("#### Output Data")

        if selected_pathway and matching_source_files and "Source_File" in outputs_df.columns:
            filtered_outputs = outputs_df[outputs_df["Source_File"].isin(matching_source_files)]
        else:
            filtered_outputs = pd.DataFrame()
        
        if not filtered_outputs.empty:
            available_locations = (
                unique_locations_df["Location_Locations"].astype(str).unique().tolist()
                if "Location_Locations" in unique_locations_df.columns
                else []
            )

            base_columns = [
                "Is reference?",
                "Flow",
                "Category",
                "Amount",
                "Unit",
                "Provider",
                "Location",
            ]
            if "Contribution Category" in filtered_outputs.columns:
                base_columns.append("Contribution Category")

            base_columns = [c for c in base_columns if c in filtered_outputs.columns]
            display_df = filtered_outputs[base_columns].copy()

            if "Location" in display_df.columns:
                display_df["Location"] = display_df["Location"].astype(str)

            if "Contribution Category" not in display_df.columns:
                display_df["Contribution Category"] = ""

            default_options = ["Electricity", "Materials", "Water Supply"]
            for idx in display_df.index:
                current_contrib = display_df.loc[idx, "Contribution Category"]
                if pd.isna(current_contrib) or str(current_contrib).strip() == "":
                    default_idx = (list(display_df.index).index(idx)) % len(default_options)
                    display_df.loc[idx, "Contribution Category"] = default_options[default_idx]
            
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
                    ),
                },
                hide_index=True,
                key="outputs_editor",
                use_container_width=True,
            )

            saved_df_outputs = edited_outputs.copy()
            
            if "Contribution Category" in saved_df_outputs.columns and "Category" in saved_df_outputs.columns:
                saved_df_outputs["Contribution Category"] = saved_df_outputs["Contribution Category"].replace("", pd.NA)
                saved_df_outputs["Contribution Category"] = saved_df_outputs["Contribution Category"].fillna(
                    saved_df_outputs["Category"]
                )
            
            # Store in session_state for mapping section
            st.session_state["saved_df_outputs"] = saved_df_outputs
            
            update_outputs = st.button("Update Output Values", key="update_outputs")
            
            if update_outputs:
                try:
                    if "Contribution Category" not in outputs_df.columns:
                        outputs_df["Contribution Category"] = ""
                    
                    filtered_indices = filtered_outputs.index.tolist()
                    
                    for i, orig_idx in enumerate(filtered_indices):
                        if i >= len(saved_df_outputs):
                            break
                        edited_row = saved_df_outputs.iloc[i]
                        
                        contrib_cat = edited_row.get("Contribution Category", "")
                        if pd.notna(contrib_cat) and str(contrib_cat).strip() and str(contrib_cat).lower() != "nan":
                            outputs_df.loc[orig_idx, "Contribution Category"] = str(contrib_cat).strip()

                        if "Amount" in edited_row:
                            outputs_df.loc[orig_idx, "Amount"] = edited_row["Amount"]
                        if "Location" in edited_row:
                            outputs_df.loc[orig_idx, "Location"] = edited_row["Location"]

                    outputs_df.to_csv(OUTPUTS_FILE, index=False)
                    st.success(f"✓ Output values saved to {OUTPUTS_FILE}")
                    st.info("💡 Your custom Contribution Category values have been saved!")
                    outputs_df = pd.read_csv(OUTPUTS_FILE)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving output values: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No output data available for the selected pathway.")
            # Use session_state if available, otherwise empty DataFrame
            saved_df_outputs = st.session_state.get("saved_df_outputs", pd.DataFrame())
    
    # Ensure saved_df_outputs is available (use session_state if not set locally)
    if "saved_df_outputs" not in locals() or saved_df_outputs.empty:
        saved_df_outputs = st.session_state.get("saved_df_outputs", pd.DataFrame())

    # -----------------------------
    # MAPPING SECTION
    # -----------------------------
    # Feature flag for semantic flow mapping (disabled by default)
    ENABLE_SEMANTIC_MAPPING = False

    # Import mapping function
    from utils.mapper import map_flows

    # Initialize mapping DataFrames
    inputs_mapping_df = pd.DataFrame()
    outputs_mapping_df = pd.DataFrame()

    # Get saved dataframes from session_state if local variables are empty
    if saved_df_inputs.empty and "saved_df_inputs" in st.session_state:
        saved_df_inputs = st.session_state["saved_df_inputs"]
    if saved_df_outputs.empty and "saved_df_outputs" in st.session_state:
        saved_df_outputs = st.session_state["saved_df_outputs"]
    
    # Map both inputs and outputs to database
    try:
        for df_name, df in [("inputs", saved_df_inputs), ("outputs", saved_df_outputs)]:
            if not df.empty:
                flow_mappings = map_flows(df, MAPPING_FILE)
                
                # Create mapping DataFrame
                mapping_records = []
                for orig_flow, mapping in flow_mappings.items():
                    record = {
                        "Original Flow": orig_flow,
                        "Mapped Flow": mapping["mapped_flow"],
                        "Amount": mapping["amount"],
                        "Unit": mapping["unit"],
                        "Category": mapping["category"]
                    }
                    # Always add Contribution Category (it's set in mapper with Category as fallback)
                    record["Contribution Category"] = mapping.get("contribution_category", mapping["category"])
                    if "is_reference" in mapping:
                        record["Is reference?"] = mapping["is_reference"]
                    mapping_records.append(record)
                mapping_df = pd.DataFrame(mapping_records)
                
                # Save mapping DataFrame
                if df_name == "inputs":
                    inputs_mapping_df = mapping_df
                else:
                    outputs_mapping_df = mapping_df
            else:
                # Keep empty DataFrame if no data
                if df_name == "inputs":
                    inputs_mapping_df = pd.DataFrame()
                else:
                    outputs_mapping_df = pd.DataFrame()
                
    except Exception as e:
        st.error(f"Error processing mapping: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    with st.expander(
        "🔗 **Input Mapping**",
        expanded=(st.session_state.get("expanded_section") == "input_mapping"),
    ):
        st.markdown("#### Inputs Mapping")
        if inputs_mapping_df.empty:
            if saved_df_inputs.empty:
                st.info("💡 **No inputs mapping available.** Please expand the '📥 Input Data' section and select a pathway with input data to generate mappings.")
            else:
                st.info("No inputs mapping available. The mapping process may have failed or there are no matching flows in the mapping file.")
        else:
            st.dataframe(
                inputs_mapping_df,
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

    with st.expander(
        "🔗 **Output Mapping**",
        expanded=(st.session_state.get("expanded_section") == "output_mapping"),
    ):
        st.markdown("#### Outputs Mapping")
        if outputs_mapping_df.empty:
            if saved_df_outputs.empty:
                st.info("💡 **No outputs mapping available.** Please expand the '📤 Output Data' section and select a pathway with output data to generate mappings.")
            else:
                st.info("No outputs mapping available. The mapping process may have failed or there are no matching flows in the mapping file.")
        else:
            st.dataframe(
                outputs_mapping_df,
                column_config={
                    "Original Flow": st.column_config.Column("Original Flow", help="Flow name from output data"),
                    "Mapped Flow": st.column_config.Column("Mapped Flow", help="Corresponding flow in database"),
                    "Amount": st.column_config.NumberColumn("Amount", help="Quantity", format="%.2f"),
                    "Unit": st.column_config.Column("Unit", help="Unit of measurement"),
                    "Category": st.column_config.Column("Category", help="Category of the flow")
                },
                hide_index=True,
                use_container_width=True
            )

    # -----------------------------
    # PATHWAY COMPARISON
    # -----------------------------
    with st.expander(
        "📊 **Pathway Comparison**",
        expanded=(st.session_state.get("expanded_section") == "pathway_comparison"),
    ):
        st.markdown("### Pathway Comparison")
        st.markdown("#### Select Pathways for Comparison (up to 3)")
        st.info("Select up to 3 pathways to compare their data side by side.")

        def get_pathway_options():
            options = []
            for pathway_name, pinfo in PATHWAY_MAPPINGS.items():
                for file_info in pinfo["files"]:
                    option_label = f"{pathway_name} - {file_info['label']}"
                    options.append(
                        {
                            "label": option_label,
                            "pathway": pathway_name,
                            "file_label": file_info["label"],
                            "source_file": file_info["source_file"],
                        }
                    )
            return options

        pathway_options_list = get_pathway_options()
        pathway_option_labels = [opt["label"] for opt in pathway_options_list]

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        selected_comparison_pathways = []

        with comp_col1:
            pathway1_comp = st.selectbox(
                "Pathway 1:",
                options=["None"] + pathway_option_labels,
                key="pathway1_comp",
            )
            if pathway1_comp != "None":
                selected_comparison_pathways.append(
                    next(opt for opt in pathway_options_list if opt["label"] == pathway1_comp)
                )

        with comp_col2:
            pathway2_comp = st.selectbox(
                "Pathway 2:",
                options=["None"] + pathway_option_labels,
                key="pathway2_comp",
            )
            if pathway2_comp != "None" and pathway2_comp not in [p["label"] for p in selected_comparison_pathways]:
                selected_comparison_pathways.append(
                    next(opt for opt in pathway_options_list if opt["label"] == pathway2_comp)
                )

        with comp_col3:
            pathway3_comp = st.selectbox(
                "Pathway 3:",
                options=["None"] + pathway_option_labels,
                key="pathway3_comp",
            )
            if pathway3_comp != "None" and pathway3_comp not in [p["label"] for p in selected_comparison_pathways]:
                selected_comparison_pathways.append(
                    next(opt for opt in pathway_options_list if opt["label"] == pathway3_comp)
                )

        if selected_comparison_pathways:
            st.markdown("---")
            st.markdown("#### Pathway Comparison Results")

            from utils.calculate import calculate_impacts

            available_source_files = (
                general_info_df["Source_File"].unique()
                if "Source_File" in general_info_df.columns
                else []
            )

            comparison_data = []

            for pathway_opt in selected_comparison_pathways:
                source_file = pathway_opt["source_file"]

                matching_files = []
                exact_matches = [sf for sf in available_source_files if sf == source_file]
                if exact_matches:
                    matching_files.extend(exact_matches)
                else:
                    partial_matches = [
                        sf
                        for sf in available_source_files
                        if source_file.lower().replace("_", " ") in sf.lower().replace("_", " ")
                        or sf.lower().replace("_", " ") in source_file.lower().replace("_", " ")
                    ]
                    matching_files.extend(partial_matches)

                if not matching_files:
                    continue

                pathway_inputs = (
                    inputs_df[inputs_df["Source_File"].isin(matching_files)]
                    if "Source_File" in inputs_df.columns
                    else pd.DataFrame()
                )
                pathway_outputs = (
                    outputs_df[outputs_df["Source_File"].isin(matching_files)]
                    if "Source_File" in outputs_df.columns
                    else pd.DataFrame()
                )

                num_input_flows = len(pathway_inputs)
                num_output_flows = len(pathway_outputs)
                total_flows = num_input_flows + num_output_flows

                total_co2_eq = 0.0

                try:
                    # inputs
                    if not pathway_inputs.empty:
                        inputs_mapping = map_flows(pathway_inputs, MAPPING_FILE)
                        tmp_inputs_map_df = pd.DataFrame(
                            [
                                {
                                    "Mapped Flow": m.get("mapped_flow", ""),
                                    "Amount": m.get("amount", None),
                                    "Unit": m.get("unit", ""),
                                    "Category": m.get("category", ""),
                                    "Contribution Category": m.get("contribution_category", m.get("category", "")),
                                }
                                for m in inputs_mapping.values()
                            ]
                        )
                        if not tmp_inputs_map_df.empty:
                            inputs_results = calculate_impacts(
                                tmp_inputs_map_df, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)"
                            )
                            if not inputs_results.empty and "Calculated Result" in inputs_results.columns:
                                total_co2_eq += float(inputs_results["Calculated Result"].sum())

                    # outputs
                    if not pathway_outputs.empty:
                        outputs_mapping = map_flows(pathway_outputs, MAPPING_FILE)
                        tmp_outputs_map_df = pd.DataFrame(
                            [
                                {
                                    "Mapped Flow": m.get("mapped_flow", ""),
                                    "Amount": m.get("amount", None),
                                    "Unit": m.get("unit", ""),
                                    "Category": m.get("category", ""),
                                    "Contribution Category": m.get("contribution_category", m.get("category", "")),
                                }
                                for m in outputs_mapping.values()
                            ]
                        )
                        if not tmp_outputs_map_df.empty:
                            try:
                                outputs_results = calculate_impacts(
                                    tmp_outputs_map_df, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)"
                                )
                                if not outputs_results.empty and "Calculated Result" in outputs_results.columns:
                                    total_co2_eq += float(outputs_results["Calculated Result"].sum())
                            except Exception as e:
                                st.warning(f"Could not calculate CO2 eq for {pathway_opt['label']}: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not process pathway {pathway_opt['label']}: {str(e)}")

                comparison_data.append(
                    {
                        "Pathway": pathway_opt["label"],
                        "Number of Flows": total_flows,
                        "CO₂ eq (kg CO₂ eq)": total_co2_eq,
                        "Total GWP (kg CO₂ eq)": total_co2_eq,
                    }
                )

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                avg_gwp = comparison_df["Total GWP (kg CO₂ eq)"].mean() if not comparison_df.empty else 0.0

                st.markdown("##### Summary Metrics")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

                with summary_col1:
                    st.metric("Number of Pathways", len(comparison_df))

                with summary_col2:
                    st.metric("Total Flows", int(comparison_df["Number of Flows"].sum()))

                with summary_col3:
                    st.metric("Average GWP", f"{avg_gwp:.3f} kg CO₂ eq")

                with summary_col4:
                    st.metric(
                        "Total GWP (All Pathways)",
                        f"{comparison_df['Total GWP (kg CO₂ eq)'].sum():.3f} kg CO₂ eq",
                    )

                st.markdown("---")
                st.markdown("##### Detailed Comparison")
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                st.session_state["selected_comparison_pathways"] = selected_comparison_pathways
                st.session_state["pathway_comparison_results"] = comparison_df
            else:
                st.warning("No comparison results available (no matching files found).")

# -----------------------------
# RIGHT COLUMN
# -----------------------------
with right_col:
    from utils.calculate import calculate_impacts
    from utils.visualize_plotly import (
        generate_impact_barchart_plotly,
        generate_impact_piechart_plotly,
        generate_impact_linechart_plotly,
    )

    # -----------------------------
    # GWP ANALYSIS
    # -----------------------------
    with st.expander(
        "🌍 **GWP Analysis**",
        expanded=(st.session_state.get("expanded_section") == "gwp_analysis"),
    ):
        st.markdown("### GWP Analysis")
    
        if st.button("Calculate GWP for All Pathways", key="gwp_calc_btn"):
            try:
                from process_lci import LCIProcessor
                
                with st.spinner("Processing all LCI data..."):
                    processor = LCIProcessor()
                    _ = processor.load_all_lci_data()
                    _ = processor.load_recipe_data()
                    gwp_results = processor.calculate_gwp()
                
                if gwp_results is not None and not gwp_results.empty:
                    st.markdown("#### Pathway GWP Results")
                    st.dataframe(gwp_results, hide_index=True, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        best_gwp = float(gwp_results.iloc[-1]["total_gwp_kgco2e"])
                        st.metric("Best Pathway GWP", f"{best_gwp:.3f} kg CO₂-eq")
                    
                    with col2:
                        worst_gwp = float(gwp_results.iloc[0]["total_gwp_kgco2e"])
                        st.metric("Worst Pathway GWP", f"{worst_gwp:.3f} kg CO₂-eq")
                    
                    with col3:
                        improvement = ((worst_gwp - best_gwp) / worst_gwp * 100) if worst_gwp > 0 else 0.0
                        st.metric("Improvement Potential", f"{improvement:.1f}%")
                    
                    csv = gwp_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download GWP Results",
                        data=csv,
                        file_name="pathway_gwp_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No GWP results available")
                
            except Exception as e:
                st.error(f"Error calculating GWP: {str(e)}")
            st.info("Make sure the LCI Excel files are in the 'input/exported LCI models/' directory")

    # -----------------------------
    # IMPACT RESULTS
    # -----------------------------
    with st.expander(
        "📈 **Impact Results**",
        expanded=(st.session_state.get("expanded_section") == "impact_results"),
    ):
        st.markdown("### Impact Results")

        # Initialize results DataFrames
        inputs_results_df = pd.DataFrame()
        outputs_results_df = pd.DataFrame()
        total_results_df = pd.DataFrame()

        # Check for semantic mapping results
        if "semantic_mapped_flows" in st.session_state:
            semantic_mapped = st.session_state["semantic_mapped_flows"]
            if semantic_mapped is not None and not semantic_mapped.empty:
                st.info("📊 Using semantic mapping results for impact calculation")
                try:
                    column_of_interest = "Carbon footprint (kg CO2 equiv.)"
                    semantic_results_df = calculate_impacts(semantic_mapped, IDEMAT_SHEET, column_of_interest)
                    if not semantic_results_df.empty:
                        semantic_results_df["Type"] = "Input (Semantic Mapping)"
                        total_results_df = pd.concat([total_results_df, semantic_results_df], ignore_index=True)
                except Exception as e:
                    st.warning(f"Could not calculate impacts for semantic mapping: {str(e)}")

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
                inputs_results_df["Type"] = "Input"

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
                outputs_results_df["Type"] = "Output"

        if not inputs_results_df.empty and not outputs_results_df.empty:
            total_results_df = pd.concat([inputs_results_df, outputs_results_df], ignore_index=True)
        elif not inputs_results_df.empty:
            total_results_df = inputs_results_df.copy()
        elif not outputs_results_df.empty:
            total_results_df = outputs_results_df.copy()
        
        if not total_results_df.empty:
            st.session_state["impact_results"] = total_results_df
            st.session_state["selected_pathway"] = selected_pathway

            if matching_source_files:
                if "Source_File" in inputs_df.columns:
                    st.session_state["filtered_inputs_for_pdf"] = inputs_df[
                        inputs_df["Source_File"].isin(matching_source_files)
                    ]
                if "Source_File" in outputs_df.columns:
                    st.session_state["filtered_outputs_for_pdf"] = outputs_df[
                        outputs_df["Source_File"].isin(matching_source_files)
                    ]

            st.markdown("#### Total Impact")
            st.dataframe(total_results_df, hide_index=True, use_container_width=True)

            if "Calculated Result" in total_results_df.columns:
                total_impact = float(total_results_df["Calculated Result"].sum())
                st.metric("Total CO₂ eq Impact", f"{total_impact:.3f} kg CO₂ eq")
        else:
            if inputs_mapping_df.empty and outputs_mapping_df.empty:
                st.info("No input or output mapping data available")
            else:
                st.warning("No impact results calculated")

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    with st.expander(
        "📊 **Visualization**",
        expanded=(st.session_state.get("expanded_section") == "visualization"),
    ):
        st.markdown("### Visualization")
        
        chart_type = st.selectbox(
            "Select chart type",
            ["Pie Chart", "Bar Chart", "Line Chart"],
            key="chart_type_selector",
        )

        if st.button("Generate Chart", key="generate_chart_btn"):
            chart_data = total_results_df if "total_results_df" in locals() and not total_results_df.empty else (
                inputs_results_df if "inputs_results_df" in locals() and not inputs_results_df.empty else outputs_results_df
            )

            if chart_data is None or chart_data.empty:
                st.warning("No data available for chart generation")
            else:
                try:
                    from utils.visualize_plotly import prepare_category_data

                    category_impacts = prepare_category_data(chart_data)
                    total_impact = float(category_impacts.sum()) if category_impacts is not None else 0.0

                    if chart_type == "Pie Chart":
                        fig, color_mapping = generate_impact_piechart_plotly(chart_data)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Bar Chart":
                        fig, color_mapping = generate_impact_barchart_plotly(chart_data)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, color_mapping = generate_impact_linechart_plotly(chart_data)
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### **Category Legend**")

                    num_cols = 3
                    cols = st.columns(num_cols)

                    if category_impacts is not None and len(category_impacts) > 0:
                        sorted_categories = category_impacts.sort_values(ascending=False)

                        for idx, (category, impact_value) in enumerate(sorted_categories.items()):
                            col_idx = idx % num_cols
                            percentage = (impact_value / total_impact * 100) if total_impact > 0 else 0.0
                            color = color_mapping.get(category, "#cccccc")

                            with cols[col_idx]:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: {color};
                                        padding: 12px;
                                        border-radius: 5px;
                                        margin-bottom: 15px;
                                        border: 2px solid #2c3e50;
                                        min-height: 80px;
                                    ">
                                        <p style="margin: 0; font-weight: bold; color: #2c3e50; font-size: 14px;">
                                            {category}
                                        </p>
                                        <p style="margin: 5px 0 0 0; color: #34495e; font-size: 12px; font-weight: 500;">
                                            {impact_value:.3f} kg CO₂ eq ({percentage:.2f}%)
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No category impact breakdown available to display.")

                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
