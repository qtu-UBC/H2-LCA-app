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

# Canadian provinces and territories for Location dropdown
CANADIAN_PROVINCES = [
    "Alberta",
    "British Columbia",
    "Manitoba",
    "New Brunswick",
    "Newfoundland and Labrador",
    "Northwest Territories",
    "Nova Scotia",
    "Nunavut",
    "Ontario",
    "Prince Edward Island",
    "Quebec",
    "Saskatchewan",
    "Yukon",
]

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


# Editable column styling: medium green background, light text (readable in light and dark mode)
EDITABLE_BG = "#43a047"
EDITABLE_TEXT = "#f5f5f5"


def _preview_table_html(df: pd.DataFrame, editable_cols: list) -> str:
    """Build HTML table with editable columns styled (background #43a047, text #f5f5f5)."""
    df = df.fillna("").astype(str)
    cols = list(df.columns)
    green_set = set(editable_cols)
    buf = ['<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">']
    buf.append("<thead><tr>")
    for c in cols:
        if c in green_set:
            style = f"background-color:{EDITABLE_BG}; color:{EDITABLE_TEXT};"
        else:
            style = ""
        label = "Type/Location" if c == "Location" else c
        buf.append(f'<th style="padding:8px; border:1px solid #ddd; text-align:left;{style}">{label}</th>')
    buf.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        buf.append("<tr>")
        for c in cols:
            if c in green_set:
                style = f"background-color:{EDITABLE_BG}; color:{EDITABLE_TEXT};"
            else:
                style = ""
            buf.append(f'<td style="padding:8px; border:1px solid #ddd;{style}">{row[c]}</td>')
        buf.append("</tr>")
    buf.append("</tbody></table>")
    return "".join(buf)


general_info_df = safe_read_csv(GENERAL_INFO_FILE)
inputs_df = safe_read_csv(INPUTS_FILE)
outputs_df = safe_read_csv(OUTPUTS_FILE)
unique_locations_df = safe_read_csv(UNIQUE_FLOWS_PROVIDERS_FILE)

# Set up the Streamlit page with full width
st.set_page_config(layout="wide")

st.title("Hydrogen Environmental Impact Calculator")

# -----------------------------
# INSTRUCTIONS (expand/collapse on click)
# -----------------------------
with st.expander("📋 **Instructions**", expanded=False):
    st.markdown("""
The app allows you to calculate environmental impact of hydrogen production pathways and test important parameters that affect the analysis. The calculator uses the Idemat open-source LCA database for background data. Follow the steps below for the calculation.

**1.** Select the H₂ production pathway (e.g., autothermal reforming) and variant (with or without carbon capture) from the drop-down menus of **Pathway File** and **Select File Variant**. This selection loads life cycle inventory (LCI) data from the openLCA collaboration server of NRC Datahub.

**2.** Select the **Input Data** button or scroll down to the Input Data panel. This panel lists inputs to the H₂ production process. The columns **Amount**, **Type/Location** and **Contribution Category** can be changed by the user. For example, these modifications can be used to test the sensitivity of electricity consumption of the H₂ production process and electricity grid mix. Contribution category allows you to categorize flows for representation of the results. When you edit a cell and press Enter (or the app re-runs), your changes are used immediately for mapping and impact in the current session. To save your changes to file so they persist, click **Update Input Values** below the table.

**3.** Select the **Output Data** button or scroll down to the Output Data panel. This panel lists outputs from the H₂ production process. The columns **Amount**, **Type/Location** and **Contribution Category** can be changed by the user. Currently, there is no allocation for by-products and all environmental impacts are allocated to hydrogen production. When you edit a cell and press Enter (or the app re-runs), your changes are used immediately for mapping and impact in the current session. To save your changes to file so they persist, click **Update Output Values** below the table.

**4.** Check that **Input Mapping** and **Output Mapping** are correct. The original LCI data from NRC Datahub openLCA collaboration server uses the ecoinvent database as background data. This app uses the open-source Idemat database as background data. The Input and Output mapping panels show the mapping between ecoinvent and Idemat flows. Please check that the mappings are correct.

**5.** Select the **Climate Change Impact** button or scroll down to the Climate Change Impact panel to see the climate change impact of each input and output. Total climate change impact is calculated as the sum of individual impacts.

**6.** Select the **Visualization** button or scroll down to the Visualization panel. Select Pie Chart and click **Generate Chart**. A figure will be generated categorized by the contribution categories entered in the Input Data and Output Data panels.

**7.** Change **Amount** and **Type/Location** in the Input and Output Data panels to evaluate sensitivities of individual inputs and outputs on the final climate change impact.
""")

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

    # Also include Source_File values from inputs/outputs that belong to this pathway
    # (avoids missing rows when data files use a slightly different or truncated string)
    def source_belongs_to_pathway(row_source: str, selected: str) -> bool:
        if not row_source or (isinstance(row_source, float) and pd.isna(row_source)):
            return False
        r = str(row_source).strip().lower().replace("_", " ")
        s = selected.strip().lower().replace("_", " ")
        if r == s:
            return True
        if len(r) >= 30 and (s.startswith(r) or r.startswith(s)):
            return True
        return s in r or r in s

    for _df in (inputs_df, outputs_df):
        if "Source_File" not in _df.columns or _df.empty:
            continue
        for val in _df["Source_File"].dropna().unique():
            if source_belongs_to_pathway(val, selected_source_file):
                matching_source_files.append(val)

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

    nav_col1, nav_col2, nav_col3 = st.columns(3)

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
        create_nav_button("📈 Climate Change Impact", "impact_results", "nav_impact")
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

        # ✅ Define filtered_inputs BEFORE using it (include all rows whose Source_File belongs to this pathway)
        if selected_pathway and "Source_File" in inputs_df.columns and not inputs_df.empty:
            mask = inputs_df["Source_File"].apply(
                lambda x: source_belongs_to_pathway(x, selected_source_file)
                if pd.notna(x) else False
            )
            filtered_inputs = inputs_df.loc[mask]
        else:
            filtered_inputs = pd.DataFrame()
        
        if not filtered_inputs.empty:
            # Type/Location dropdown: Canadian provinces/territories + any existing values in data
            existing_locations = (
                filtered_inputs["Location"].dropna().astype(str).str.strip().unique().tolist()
                if "Location" in filtered_inputs.columns
                else []
            )
            available_locations = list(dict.fromkeys(CANADIAN_PROVINCES + [x for x in existing_locations if x and x not in CANADIAN_PROVINCES]))

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

            editable_cols = ["Amount", "Location", "Contribution Category"]

            edited_inputs = st.data_editor(
                display_df,
                column_config={
                    "Flow": st.column_config.TextColumn("Flow", disabled=True),
                    "Unit": st.column_config.TextColumn("Unit", disabled=True),
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value",
                        step=0.01,
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Type/Location",
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
            )

            st.caption("Editable columns (Amount, Type/Location, Contribution Category). Changes apply to mapping and impact immediately; click **Update Input Values** below to save to file.")
            st.markdown(
                _preview_table_html(display_df, editable_cols),
                unsafe_allow_html=True,
            )

            saved_df_inputs = edited_inputs.copy()

            if "Contribution Category" in saved_df_inputs.columns and "Category" in saved_df_inputs.columns:
                saved_df_inputs["Contribution Category"] = saved_df_inputs["Contribution Category"].replace("", pd.NA)
                saved_df_inputs["Contribution Category"] = saved_df_inputs["Contribution Category"].fillna(
                    saved_df_inputs["Category"]
                )

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

        if selected_pathway and "Source_File" in outputs_df.columns and not outputs_df.empty:
            mask_out = outputs_df["Source_File"].apply(
                lambda x: source_belongs_to_pathway(x, selected_source_file)
                if pd.notna(x) else False
            )
            filtered_outputs = outputs_df.loc[mask_out]
        else:
            filtered_outputs = pd.DataFrame()

        if not filtered_outputs.empty:
            existing_out = (
                filtered_outputs["Location"].dropna().astype(str).str.strip().unique().tolist()
                if "Location" in filtered_outputs.columns else []
            )
            available_locations = list(dict.fromkeys(CANADIAN_PROVINCES + [x for x in existing_out if x and x not in CANADIAN_PROVINCES]))

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

            editable_cols = ["Amount", "Location", "Contribution Category"]

            edited_outputs = st.data_editor(
                display_df,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value",
                        step=0.01,
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Type/Location",
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
            )

            st.caption("Editable columns (Amount, Type/Location, Contribution Category). Changes apply to mapping and impact immediately; click **Update Output Values** below to save to file.")
            st.markdown(
                _preview_table_html(display_df, editable_cols),
                unsafe_allow_html=True,
            )

            saved_df_outputs = edited_outputs.copy()

            if "Contribution Category" in saved_df_outputs.columns and "Category" in saved_df_outputs.columns:
                saved_df_outputs["Contribution Category"] = saved_df_outputs["Contribution Category"].replace("", pd.NA)
                saved_df_outputs["Contribution Category"] = saved_df_outputs["Contribution Category"].fillna(
                    saved_df_outputs["Category"]
                )

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
    # PATHWAY COMPARISON (commented out — uses same logic as Climate Change Impact for consistency)
    # Uncomment to show comparison of up to 3 pathways; uses compute_pathway_impact (CSV + map_flows + IDEMAT).
    # -----------------------------
    # from utils.calculate import compute_pathway_impact
    # _available = general_info_df["Source_File"].unique().tolist() if "Source_File" in general_info_df.columns else []
    # _pathway_opts = []
    # for pname, pinfo in PATHWAY_MAPPINGS.items():
    #     for f in pinfo["files"]:
    #         _pathway_opts.append({"label": f"{pname} – {f['label']}", "source_file": f["source_file"]})
    # _labels = [o["label"] for o in _pathway_opts]
    # _sel1 = st.selectbox("Pathway 1", ["None"] + _labels, key="pc1")
    # _sel2 = st.selectbox("Pathway 2", ["None"] + _labels, key="pc2")
    # _sel3 = st.selectbox("Pathway 3", ["None"] + _labels, key="pc3")
    # _selected = [o for s in [_sel1, _sel2, _sel3] if s != "None" and (o := next((x for x in _pathway_opts if x["label"] == s), None))]
    # if _selected:
    #     _rows = []
    #     for o in _selected:
    #         _sf = o["source_file"]
    #         _exact = [x for x in _available if x == _sf]
    #         _match = _exact if _exact else [x for x in _available if _sf.lower().replace("_", " ") in x.lower().replace("_", " ") or x.lower().replace("_", " ") in _sf.lower().replace("_", " ")]
    #         _total = compute_pathway_impact(inputs_df, outputs_df, _match, MAPPING_FILE, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)")
    #         _rows.append({"Pathway": o["label"], "Climate Change Impact (kg CO₂ eq./kg H2)": _total})
    #     st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

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

    # If still empty but we have a pathway selected, build from CSV so mapping can run without opening Input/Output Data first
    # Match using Source_File from the data files (inputs_df/outputs_df)
    if saved_df_inputs.empty and selected_pathway and "Source_File" in inputs_df.columns and not inputs_df.empty:
        all_sources = inputs_df["Source_File"].dropna().astype(str).unique()
        sel_norm = selected_source_file.lower().replace("_", " ")
        matching = [
            s for s in all_sources
            if s == selected_source_file
            or sel_norm in s.lower().replace("_", " ")
            or s.lower().replace("_", " ") in sel_norm
        ]
        if not matching and matching_source_files:
            matching = [s for s in all_sources if any(m in s for m in matching_source_files)]
        if not matching:
            # Fallback: match by pathway name keyword in Source_File
            kw = selected_pathway.lower().replace(" ", "")
            if "autothermal" in kw or "reforming" in kw:
                kw = "autothermal"
            elif "pem" in kw or "electrolysis" in kw:
                kw = "pem"
            elif "biomass" in kw or "gasification" in kw:
                kw = "biomass"
            else:
                kw = selected_pathway.lower()[:20]
            matching = [s for s in all_sources if kw in s.lower().replace("_", " ").replace(" ", "")]
            if not matching:
                matching = list(all_sources)
        filtered = inputs_df[inputs_df["Source_File"].astype(str).isin(matching)]
        if not filtered.empty:
            base_cols = [c for c in ["Flow", "Category", "Amount", "Unit", "Provider", "Location", "Contribution Category"] if c in filtered.columns]
            saved_df_inputs = filtered[base_cols].copy()
            st.session_state["saved_df_inputs"] = saved_df_inputs
    if saved_df_outputs.empty and selected_pathway and "Source_File" in outputs_df.columns and not outputs_df.empty:
        all_sources = outputs_df["Source_File"].dropna().astype(str).unique()
        sel_norm = selected_source_file.lower().replace("_", " ")
        matching = [
            s for s in all_sources
            if s == selected_source_file
            or sel_norm in s.lower().replace("_", " ")
            or s.lower().replace("_", " ") in sel_norm
        ]
        if not matching and matching_source_files:
            matching = [s for s in all_sources if any(m in s for m in matching_source_files)]
        if not matching:
            kw = selected_pathway.lower().replace(" ", "")
            if "autothermal" in kw or "reforming" in kw:
                kw = "autothermal"
            elif "pem" in kw or "electrolysis" in kw:
                kw = "pem"
            elif "biomass" in kw or "gasification" in kw:
                kw = "biomass"
            else:
                kw = selected_pathway.lower()[:20]
            matching = [s for s in all_sources if kw in s.lower().replace("_", " ").replace(" ", "")]
            if not matching:
                matching = list(all_sources)
        filtered = outputs_df[outputs_df["Source_File"].astype(str).isin(matching)]
        if not filtered.empty:
            base_cols = [c for c in ["Flow", "Category", "Amount", "Unit", "Provider", "Location", "Contribution Category", "Is reference?"] if c in filtered.columns]
            saved_df_outputs = filtered[base_cols].copy()
            st.session_state["saved_df_outputs"] = saved_df_outputs

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
            )

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
    # GWP ANALYSIS (commented out — uses same logic as Climate Change Impact for consistency)
    # Uncomment to show GWP for all pathways; uses compute_pathway_impact (CSV + map_flows + IDEMAT), not process_lci/ReCiPe.
    # -----------------------------
    # from utils.calculate import compute_pathway_impact
    # _available_sf = general_info_df["Source_File"].unique().tolist() if "Source_File" in general_info_df.columns else []
    # if st.button("Calculate GWP for All Pathways (consistent with Climate Change Impact)", key="gwp_btn"):
    #     _gwp_rows = []
    #     for pname, pinfo in PATHWAY_MAPPINGS.items():
    #         for f in pinfo["files"]:
    #             _sf = f["source_file"]
    #             _exact = [x for x in _available_sf if x == _sf]
    #             _match = _exact if _exact else [x for x in _available_sf if _sf.lower().replace("_", " ") in x.lower().replace("_", " ") or x.lower().replace("_", " ") in _sf.lower().replace("_", " ")]
    #             _total = compute_pathway_impact(inputs_df, outputs_df, _match, MAPPING_FILE, IDEMAT_SHEET, "Carbon footprint (kg CO2 equiv.)")
    #             _gwp_rows.append({"Pathway": f"{pname} – {f['label']}", "Climate Change Impact (kg CO₂ eq./kg H2)": _total})
    #     _gwp_df = pd.DataFrame(_gwp_rows)
    #     if not _gwp_df.empty:
    #         st.dataframe(_gwp_df.sort_values("Climate Change Impact (kg CO₂ eq./kg H2)", ascending=False), hide_index=True, use_container_width=True)

    # -----------------------------
    # CLIMATE CHANGE IMPACT
    # -----------------------------
    with st.expander(
        "📈 **Climate Change Impact**",
        expanded=(st.session_state.get("expanded_section") == "impact_results"),
    ):
        st.markdown("### Climate Change Impact")

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
            col_config = {"Calculated Result": st.column_config.NumberColumn("Calculated Result (kg CO2 eq.)", format="%.3f")}
            if "Note" in total_results_df.columns:
                col_config["Note"] = st.column_config.TextColumn("Note", help="Reason for 0 or conversion used")
            st.dataframe(
                total_results_df,
                hide_index=True,
                column_config=col_config,
            )

            if "Calculated Result" in total_results_df.columns:
                total_impact = float(total_results_df["Calculated Result"].sum())
                st.metric("Climate Change Impact per 1 kg of Hydrogen", f"{total_impact:.3f} kg CO₂ eq./kg H2")
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
            # Get chart data from session_state (stored in Impact Results section)
            chart_data = st.session_state.get("impact_results", pd.DataFrame())

            if chart_data is None or chart_data.empty:
                st.warning("No data available for chart generation")
            else:
                try:
                    from utils.visualize_plotly import prepare_category_data

                    category_impacts = prepare_category_data(chart_data)
                    total_impact = float(category_impacts.sum()) if category_impacts is not None else 0.0

                    # Plotly config: use config dict to avoid deprecated kwargs warning
                    plotly_config = {"displayModeBar": True, "responsive": True}
                    if chart_type == "Pie Chart":
                        fig, color_mapping = generate_impact_piechart_plotly(chart_data)
                        st.plotly_chart(fig, config=plotly_config, use_container_width=True)
                    elif chart_type == "Bar Chart":
                        fig, color_mapping = generate_impact_barchart_plotly(chart_data)
                        st.plotly_chart(fig, config=plotly_config, use_container_width=True)
                    else:
                        fig, color_mapping = generate_impact_linechart_plotly(chart_data)
                        st.plotly_chart(fig, config=plotly_config, use_container_width=True)

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
                                            {impact_value:.3f} kg CO₂ eq./kg H2 ({percentage:.2f}%)
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
