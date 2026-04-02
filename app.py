import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path
from typing import Optional

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

# Header strings only (EN/FR); body copy (e.g. instructions) stays English unless keyed here.
UI_HEADERS = {
    "en": {
        "skip_main": "Skip to main content",
        "main_landmark": "Start of main content",
        "title": "Hydrogen Environmental Impact Calculator",
        "instructions": "📋 **Instructions**",
        "pathway_file": "Pathway File:",
        "file_variant": "Select File Variant:",
        "sidebar_jump": "Jump to section",
        "nav_inputs": "Input Data",
        "nav_outputs": "Output Data",
        "nav_inmap": "Input Mapping",
        "nav_outmap": "Output Mapping",
        "nav_impact": "Climate Change Impact",
        "nav_viz": "Visualization",
        "nav_help_inputs": "Open or close the inputs table (editable amounts and locations).",
        "nav_help_outputs": "Open or close the outputs table.",
        "nav_help_inmap": "Review how input flows map to the Idemat database.",
        "nav_help_outmap": "Review how output flows map to the Idemat database.",
        "nav_help_impact": "View climate change impacts per flow and total per kg H₂.",
        "nav_help_viz": "Generate pie, bar, or line charts from impact results.",
        "exp_inputs": "📥 **Input Data**",
        "exp_outputs": "📤 **Output Data**",
        "exp_inmap": "🔗 **Input Mapping**",
        "exp_outmap": "🔗 **Output Mapping**",
        "exp_impact": "📈 **Climate Change Impact**",
        "exp_viz": "📊 **Visualization**",
        "chart_type": "Chart type",
        "generate_chart": "Generate chart",
        "category_legend": "Category legend",
        "metric_h2": "Climate change impact per 1 kg of hydrogen",
    },
    "fr": {
        "skip_main": "Aller au contenu principal",
        "main_landmark": "Début du contenu principal",
        "title": "Calculateur d'impacts environnementaux de l'hydrogène",
        "instructions": "📋 **Instructions**",
        "pathway_file": "Filière de production :",
        "file_variant": "Variante du scénario :",
        "sidebar_jump": "Accéder à une section",
        "nav_inputs": "Données d'entrée",
        "nav_outputs": "Données de sortie",
        "nav_inmap": "Correspondance des intrants",
        "nav_outmap": "Correspondance des extrants",
        "nav_impact": "Changement climatique",
        "nav_viz": "Visualisation",
        "nav_help_inputs": "Ouvrir ou fermer le tableau des intrants (quantités et lieux modifiables).",
        "nav_help_outputs": "Ouvrir ou fermer le tableau des extrants.",
        "nav_help_inmap": "Contrôler la mise en correspondance des flux entrants avec la base de données Idemat.",
        "nav_help_outmap": "Contrôler la mise en correspondance des flux sortants avec la base de données Idemat.",
        "nav_help_impact": "Consulter l'impact par flux et le total par kilogramme de H₂.",
        "nav_help_viz": "Produire des graphiques (diagrammes circulaires, histogrammes, courbes) à partir des résultats.",
        "exp_inputs": "📥 **Données d'entrée**",
        "exp_outputs": "📤 **Données de sortie**",
        "exp_inmap": "🔗 **Correspondance des intrants**",
        "exp_outmap": "🔗 **Correspondance des extrants**",
        "exp_impact": "📈 **Changement climatique**",
        "exp_viz": "📊 **Visualisation**",
        "chart_type": "Type de graphique",
        "generate_chart": "Générer le graphique",
        "category_legend": "Légende des catégories",
        "metric_h2": "Impact climatique par kilogramme d'hydrogène",
    },
}

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


def _column_width_pct(col_name: str, all_cols: list) -> float:
    """Approximate column share (sums to ~100) for preview table layout."""
    weights = {
        "Flow": 20.0,
        "Category": 12.0,
        "Amount": 10.0,
        "Unit": 8.0,
        "Provider": 12.0,
        "Location": 14.0,
        "Contribution Category": 16.0,
        "Is reference?": 8.0,
    }
    raw = [weights.get(c, 10.0) for c in all_cols]
    s = sum(raw)
    return 100.0 * (weights.get(col_name, 10.0) / s) if s else 100.0 / max(len(all_cols), 1)


def _preview_table_html(df: pd.DataFrame, editable_cols: list) -> str:
    """Build HTML table with percentage-based column widths; high-contrast text."""
    df = df.fillna("").astype(str)
    cols = list(df.columns)
    green_set = set(editable_cols)
    buf = ['<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:0.9rem;">']
    buf.append("<colgroup>")
    for c in cols:
        pct = _column_width_pct(c, cols)
        buf.append(f'<col style="width:{pct:.1f}%;" />')
    buf.append("</colgroup>")
    buf.append("<thead><tr>")
    for c in cols:
        if c in green_set:
            style = f"background-color:{EDITABLE_BG}; color:{EDITABLE_TEXT};"
        else:
            style = "color:#000000;"
        label = "Type/Location" if c == "Location" else c
        buf.append(f'<th style="padding:8px;border:1px solid #333;text-align:left;{style}">{label}</th>')
    buf.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        buf.append("<tr>")
        for c in cols:
            if c in green_set:
                style = f"background-color:{EDITABLE_BG}; color:{EDITABLE_TEXT};"
            else:
                style = "color:#000000;"
            buf.append(f'<td style="padding:8px;border:1px solid #333;{style}">{row[c]}</td>')
        buf.append("</tr>")
    buf.append("</tbody></table>")
    return "".join(buf)


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


def hydrate_saved_from_csv(
    df: pd.DataFrame,
    selected_pathway: str,
    selected_source_file: str,
    matching_source_files: list,
    base_cols: list,
) -> Optional[pd.DataFrame]:
    """Build saved dataframe from CSV when session copy is empty (same logic for inputs and outputs)."""
    if not selected_pathway or "Source_File" not in df.columns or df.empty:
        return None
    all_sources = df["Source_File"].dropna().astype(str).unique()
    sel_norm = selected_source_file.lower().replace("_", " ")
    matching = [
        s
        for s in all_sources
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
    filtered = df[df["Source_File"].astype(str).isin(matching)]
    if filtered.empty:
        return None
    use_cols = [c for c in base_cols if c in filtered.columns]
    return filtered[use_cols].copy()


def mapping_dataframe_column_config(flow_context: str) -> dict:
    """Shared column_config for input and output mapping tables (removes duplication)."""
    return {
        "Original Flow": st.column_config.Column("Original Flow", help=f"Flow name from {flow_context}"),
        "Mapped Flow": st.column_config.Column("Mapped Flow", help="Corresponding flow in database"),
        "Amount": st.column_config.NumberColumn("Amount", help="Quantity", format="%.2f"),
        "Unit": st.column_config.Column("Unit", help="Unit of measurement"),
        "Category": st.column_config.Column("Category", help="Category of the flow"),
    }


def default_contribution_category(flow_val) -> str:
    """Default contribution category by flow type: electricity -> Electricity, water -> Water Supply, else Materials."""
    if pd.isna(flow_val):
        return "Materials"
    s = str(flow_val).strip().lower()
    if "electricity" in s:
        return "Electricity"
    if "water" in s or "deionised" in s or "deionized" in s:
        return "Water Supply"
    return "Materials"


def apply_contribution_category_defaults(display_df: pd.DataFrame) -> None:
    """Fill empty or generic Contribution Category from flow name (mutates display_df in place)."""
    if "Contribution Category" not in display_df.columns or "Flow" not in display_df.columns:
        return
    default_options = ("Electricity", "Materials", "Water Supply")
    for idx in display_df.index:
        current = display_df.loc[idx, "Contribution Category"]
        current_str = "" if pd.isna(current) else str(current).strip()
        suggested = default_contribution_category(display_df.loc[idx, "Flow"])
        if not current_str or (current_str in default_options and current_str != suggested):
            display_df.loc[idx, "Contribution Category"] = suggested


def create_section_nav_button(label: str, section_key: str, button_key: str, help_text: str) -> None:
    """Toggle which main-panel expander is open; used from the sidebar."""
    current_section = st.session_state.get("expanded_section", None)
    is_active = current_section == section_key
    button_type = "primary" if is_active else "secondary"
    if st.button(
        label,
        use_container_width=True,
        key=button_key,
        type=button_type,
        help=help_text,
    ):
        if is_active:
            st.session_state["expanded_section"] = None
        else:
            st.session_state["expanded_section"] = section_key
        st.rerun()


general_info_df = safe_read_csv(GENERAL_INFO_FILE)
# Load from file only. The app NEVER writes to inputs.csv or outputs.csv; all edits are session-only.
inputs_df = safe_read_csv(INPUTS_FILE)
outputs_df = safe_read_csv(OUTPUTS_FILE)
unique_locations_df = safe_read_csv(UNIQUE_FLOWS_PROVIDERS_FILE)

# Reset counter: increment on "Reset to default" so data_editor re-initializes with file data
if "inputs_editor_reset_key" not in st.session_state:
    st.session_state["inputs_editor_reset_key"] = 0
if "outputs_editor_reset_key" not in st.session_state:
    st.session_state["outputs_editor_reset_key"] = 0

# New session only (reload/new tab): clear saved data and bump editor keys so table shows file values
if "_session_initialized" not in st.session_state:
    for key in ("saved_df_inputs", "saved_df_outputs", "saved_inputs_pathway", "saved_outputs_pathway"):
        st.session_state.pop(key, None)
    st.session_state["inputs_editor_reset_key"] = st.session_state.get("inputs_editor_reset_key", 0) + 1
    st.session_state["outputs_editor_reset_key"] = st.session_state.get("outputs_editor_reset_key", 0) + 1
    st.session_state["_session_initialized"] = True

# Set up the Streamlit page with full width and accessible title
st.set_page_config(
    layout="wide",
    page_title="Hydrogen Environmental Impact Calculator",
    initial_sidebar_state="expanded",
)

if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"

T = UI_HEADERS[st.session_state.get("ui_lang", "en")]
st.session_state.setdefault("expanded_section", "input_data")

with st.sidebar:
    st.caption("English / Français — section titles & navigation only")
    sb1, sb2 = st.columns(2)
    with sb1:
        if st.button(
            "English",
            key="set_lang_en",
            use_container_width=True,
            help="Show section titles and navigation in English",
        ):
            st.session_state["ui_lang"] = "en"
            st.rerun()
    with sb2:
        if st.button(
            "Français",
            key="set_lang_fr",
            use_container_width=True,
            help="Afficher les titres de section et la navigation en français",
        ):
            st.session_state["ui_lang"] = "fr"
            st.rerun()

    st.markdown("---")
    st.caption(T["sidebar_jump"])
    with st.container(border=True):
        create_section_nav_button(T["nav_inputs"], "input_data", "nav_input_data", T["nav_help_inputs"])
        create_section_nav_button(T["nav_outputs"], "output_data", "nav_output_data", T["nav_help_outputs"])
        create_section_nav_button(T["nav_inmap"], "input_mapping", "nav_input_mapping", T["nav_help_inmap"])
        create_section_nav_button(T["nav_outmap"], "output_mapping", "nav_output_mapping", T["nav_help_outmap"])
        create_section_nav_button(T["nav_impact"], "impact_results", "nav_impact", T["nav_help_impact"])
        create_section_nav_button(T["nav_viz"], "visualization", "nav_viz", T["nav_help_viz"])

# Remove Streamlit's default "Running..." animation (top-right), responsive layout, high-contrast text
st.markdown(
    """
    <style>
    [data-testid="stStatusWidget"],
    div[class*="stStatusWidget"],
    [aria-label="running"],
    [aria-label="Running"] {
        display: none !important;
    }

    .block-container {
        max-width: min(100%, 1600px) !important;
        padding-left: clamp(0.75rem, 2.5vw, 2rem) !important;
        padding-right: clamp(0.75rem, 2.5vw, 2rem) !important;
    }
    @media (max-width: 1024px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }
    }

    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li,
    [data-testid="stAppViewContainer"] .stMarkdown span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stCaption"],
    [data-testid="stAppViewContainer"] .stAlert p,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    .skip-link {
        position: absolute;
        left: -9999px;
        z-index: 9999;
        padding: 0.75rem 1rem;
        background: #000000;
        color: #ffffff;
        font-weight: bold;
        text-decoration: none;
        border-radius: 0.25rem;
    }
    .skip-link:focus {
        left: 0.5rem;
        top: 0.5rem;
    }

    button:focus-visible,
    [role="button"]:focus-visible,
    a:focus-visible,
    input:focus-visible,
    select:focus-visible,
    [role="tab"]:focus-visible {
        outline: 3px solid #000000;
        outline-offset: 2px;
    }

    /* Section-jump nav: bordered strip — label size/weight like HTML <h2> */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button {
        font-size: clamp(1.2rem, 2.5vw, 1.7rem) !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
        min-height: 2.85rem !important;
        letter-spacing: -0.015em;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button[kind="secondary"] {
        color: #000000 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: #ffffff !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    /* Slightly smaller type in the narrow sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button {
        font-size: clamp(0.9rem, 1.8vw, 1.2rem) !important;
        min-height: 2.35rem !important;
    }

    /* Main content: expander labels = section titles — size/weight like HTML <h2> (Streamlit uses .main) */
    .main [data-testid="stExpander"] summary {
        font-size: clamp(1.2rem, 2.4vw, 1.65rem) !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
        color: #000000 !important;
    }
    .main [data-testid="stExpander"] summary p {
        font-size: inherit !important;
        font-weight: inherit !important;
        color: inherit !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f'<a href="#main-content" class="skip-link">{T["skip_main"]}</a>',
    unsafe_allow_html=True,
)

st.title(T["title"])

st.markdown(
    f'<span id="main-content" tabindex="-1" aria-label="{T["main_landmark"]}"></span>',
    unsafe_allow_html=True,
)

# -----------------------------
# INSTRUCTIONS (expand/collapse on click)
# -----------------------------
with st.expander(T["instructions"], expanded=False):
    st.markdown("""
The app allows you to calculate environmental impact of hydrogen production pathways and test important parameters that affect the analysis. The calculator uses the Idemat open-source LCA database for background data. Follow the steps below for the calculation.

**1.** Select the H₂ production pathway (e.g., autothermal reforming) and variant (with or without carbon capture) from the drop-down menus of **Pathway File** and **Select File Variant**. This selection loads life cycle inventory (LCI) data from the openLCA collaboration server of NRC Datahub.

**2.** Use the **Input Data** control or scroll to the inputs panel. This panel lists inputs to the H₂ production process. The columns **Amount**, **Type/Location** and **Contribution Category** can be changed by the user. Click **Update Input Values** to apply edits for this session only (the CSV file is never modified). Click **Reset to default values** to discard session edits and reload from file. Reloading the app loads the original files.

**3.** Use the **Output Data** control or scroll to the outputs panel. This panel lists outputs from the H₂ production process. The columns **Amount**, **Type/Location** and **Contribution Category** can be changed by the user. Click **Update Output Values** to apply edits for this session only (the CSV file is never modified). Click **Reset to default values** to discard session edits and reload from file. Reloading the app loads the original files.

**4.** Check that **Input Mapping** and **Output Mapping** are correct. The original LCI data from NRC Datahub openLCA collaboration server uses the ecoinvent database as background data. This app uses the open-source Idemat database as background data. The Input and Output mapping panels show the mapping between ecoinvent and Idemat flows. Please check that the mappings are correct.

**5.** Use **Climate Change Impact** (or scroll) to see the climate change impact of each input and output. Total climate change impact is calculated as the sum of individual impacts.

**6.** Use **Visualization** (or scroll). Select a chart type, then use the chart generation button. A figure will be generated categorized by the contribution categories from the inputs and outputs panels.

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

# Single-column layout (full width / tablet-friendly stacking)
# Initialize these so they always exist (prevents NameError)
saved_df_inputs = pd.DataFrame()
saved_df_outputs = pd.DataFrame()
inputs_mapping_df = pd.DataFrame()
outputs_mapping_df = pd.DataFrame()

# -----------------------------
# Main content (pathway, tables, mappings, impacts, charts)
# -----------------------------
with st.container():
    pathway_options = list(PATHWAY_MAPPINGS.keys())
    selected_pathway = st.selectbox(
        T["pathway_file"],
        options=pathway_options,
        help="Select a pathway to view its data",
    )

    pathway_info = PATHWAY_MAPPINGS[selected_pathway]
    pathway_files = pathway_info["files"]

    selected_source_file = None
    if len(pathway_files) > 1:
        file_labels = [f["label"] for f in pathway_files]
        selected_file_label = st.selectbox(
            T["file_variant"],
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
    for _df in (inputs_df, outputs_df):
        if "Source_File" not in _df.columns or _df.empty:
            continue
        for val in _df["Source_File"].dropna().unique():
            if source_belongs_to_pathway(val, selected_source_file):
                matching_source_files.append(val)

    matching_source_files = list(dict.fromkeys(matching_source_files))

    st.markdown(
        f'<p style="color: #000000; font-size: 0.95em; margin-top: 10px; font-weight: 500;">{pathway_info["description"]}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -----------------------------
    # INPUT DATA
    # -----------------------------
    with st.expander(
        T["exp_inputs"],
        expanded=(st.session_state.get("expanded_section") == "input_data"),
    ):
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
            # Type/Location: Canada + Canadian provinces/territories + any existing values (dropdown is searchable – type to filter, Enter to select)
            existing_locations = (
                filtered_inputs["Location"].dropna().astype(str).str.strip().unique().tolist()
                if "Location" in filtered_inputs.columns
                else []
            )
            extra = [x for x in existing_locations if x and x not in CANADIAN_PROVINCES and x != "Canada"]
            available_locations = list(dict.fromkeys(["Canada"] + CANADIAN_PROVINCES + extra))

            base_columns = ["Flow", "Category", "Amount", "Unit", "Provider", "Location"]
            if "Contribution Category" in filtered_inputs.columns:
                base_columns.append("Contribution Category")

            base_columns = [c for c in base_columns if c in filtered_inputs.columns]
            # Use saved session data when available (so edits and Update persist in table); else file (new session / Reset)
            _saved = st.session_state.get("saved_df_inputs", pd.DataFrame())
            _path = st.session_state.get("saved_inputs_pathway")
            if not _saved.empty and _path == selected_source_file and set(base_columns).issubset(set(_saved.columns)):
                display_df = _saved[[c for c in base_columns if c in _saved.columns]].copy()
            else:
                display_df = filtered_inputs[base_columns].copy()

            if "Location" in display_df.columns:
                display_df["Location"] = display_df["Location"].astype(str)

            if "Contribution Category" not in display_df.columns:
                display_df["Contribution Category"] = ""

            apply_contribution_category_defaults(display_df)

            editable_cols = ["Amount", "Location", "Contribution Category"]

            edited_inputs = st.data_editor(
                display_df,
                column_config={
                    "Flow": st.column_config.TextColumn("Flow", disabled=True, width="large"),
                    "Unit": st.column_config.TextColumn("Unit", disabled=True, width="small"),
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value",
                        step=0.01,
                        width="small",
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Type/Location",
                        help="Select or type to search: Canada, provinces, or existing values; press Enter to apply",
                        options=available_locations,
                        default="",
                        width="medium",
                    ),
                    "Contribution Category": st.column_config.TextColumn(
                        "Contribution Category",
                        help="Edit the contribution category. Defaults: Electricity, Materials, Water Supply. You can type any custom category.",
                        default="",
                        width="medium",
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key="inputs_editor_" + str(st.session_state.get("inputs_editor_reset_key", 0)),
            )

            st.caption(
                "Editable columns: Amount, Type/Location, Contribution Category. The CSV file is never modified. "
                "Update Input Values applies edits for this session only; Reset to default values reloads from the file."
            )
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
            st.session_state["saved_inputs_pathway"] = selected_source_file

            update_inputs = st.button(
                "Update Input Values",
                key="update_inputs",
                use_container_width=True,
                help="Apply the current table values for this session only. The CSV file on disk is never modified.",
            )
            reset_inputs = st.button(
                "Reset to default values",
                key="reset_inputs",
                use_container_width=True,
                help="Discard session edits and reload the input table from the original CSV file.",
            )

            if reset_inputs:
                st.session_state["inputs_editor_reset_key"] = st.session_state.get("inputs_editor_reset_key", 0) + 1
                st.rerun()

            if update_inputs:
                # Session only: do NOT write to file; table will show file again on next run (reload = original)
                st.success("✓ Input values applied for this session. Reload or next run will show original file values.")
                st.rerun()
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
        T["exp_outputs"],
        expanded=(st.session_state.get("expanded_section") == "output_data"),
    ):
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
            extra_out = [x for x in existing_out if x and x not in CANADIAN_PROVINCES and x != "Canada"]
            available_locations = list(dict.fromkeys(["Canada"] + CANADIAN_PROVINCES + extra_out))

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
            # Use saved session data when available (so edits and Update persist in table); else file (new session / Reset)
            _saved_out = st.session_state.get("saved_df_outputs", pd.DataFrame())
            _path_out = st.session_state.get("saved_outputs_pathway")
            if not _saved_out.empty and _path_out == selected_source_file and set(base_columns).issubset(set(_saved_out.columns)):
                display_df = _saved_out[[c for c in base_columns if c in _saved_out.columns]].copy()
            else:
                display_df = filtered_outputs[base_columns].copy()

            if "Location" in display_df.columns:
                display_df["Location"] = display_df["Location"].astype(str)

            if "Contribution Category" not in display_df.columns:
                display_df["Contribution Category"] = ""

            apply_contribution_category_defaults(display_df)

            editable_cols = ["Amount", "Location", "Contribution Category"]

            edited_outputs = st.data_editor(
                display_df,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        help="Edit the amount value",
                        step=0.01,
                        width="small",
                    ),
                    "Location": st.column_config.SelectboxColumn(
                        "Type/Location",
                        help="Select or type to search: Canada, provinces, or existing values; press Enter to apply",
                        options=available_locations,
                        default="",
                        width="medium",
                    ),
                    "Contribution Category": st.column_config.TextColumn(
                        "Contribution Category",
                        help="Edit the contribution category. Defaults: Electricity, Materials, Water Supply. You can type any custom category.",
                        default="",
                        width="medium",
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key="outputs_editor_" + str(st.session_state.get("outputs_editor_reset_key", 0)),
            )

            st.caption(
                "Editable columns: Amount, Type/Location, Contribution Category. The CSV file is never modified. "
                "Update Output Values applies edits for this session only; Reset to default values reloads from the file."
            )
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
            st.session_state["saved_outputs_pathway"] = selected_source_file

            update_outputs = st.button(
                "Update Output Values",
                key="update_outputs",
                use_container_width=True,
                help="Apply the current table values for this session only. The CSV file on disk is never modified.",
            )
            reset_outputs = st.button(
                "Reset to default values",
                key="reset_outputs",
                use_container_width=True,
                help="Discard session edits and reload the output table from the original CSV file.",
            )

            if reset_outputs:
                st.session_state["outputs_editor_reset_key"] = st.session_state.get("outputs_editor_reset_key", 0) + 1
                st.rerun()

            if update_outputs:
                # Session only: do NOT write to file; table will show file again on next run (reload = original)
                st.success("✓ Output values applied for this session. Reload or next run will show original file values.")
                st.rerun()
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

    # If still empty, build from CSV so mapping can run without opening Input/Output Data first
    if saved_df_inputs.empty:
        _hydrated_in = hydrate_saved_from_csv(
            inputs_df,
            selected_pathway,
            selected_source_file,
            matching_source_files,
            ["Flow", "Category", "Amount", "Unit", "Provider", "Location", "Contribution Category"],
        )
        if _hydrated_in is not None:
            saved_df_inputs = _hydrated_in
            st.session_state["saved_df_inputs"] = saved_df_inputs
    if saved_df_outputs.empty:
        _hydrated_out = hydrate_saved_from_csv(
            outputs_df,
            selected_pathway,
            selected_source_file,
            matching_source_files,
            [
                "Is reference?",
                "Flow",
                "Category",
                "Amount",
                "Unit",
                "Provider",
                "Location",
                "Contribution Category",
            ],
        )
        if _hydrated_out is not None:
            saved_df_outputs = _hydrated_out
            st.session_state["saved_df_outputs"] = saved_df_outputs

    # Map both inputs and outputs to database (show wait message instead of default animation)
    try:
        with st.spinner("Please wait. Loading mappings…"):
            for df_name, df in [("inputs", saved_df_inputs), ("outputs", saved_df_outputs)]:
                if not df.empty:
                    flow_mappings = map_flows(df, MAPPING_FILE)
                    mapping_records = []
                    for orig_flow, mapping in flow_mappings.items():
                        record = {
                            "Original Flow": orig_flow,
                            "Mapped Flow": mapping["mapped_flow"],
                            "Amount": mapping["amount"],
                            "Unit": mapping["unit"],
                            "Category": mapping["category"]
                        }
                        record["Contribution Category"] = mapping.get("contribution_category", mapping["category"])
                        if "is_reference" in mapping:
                            record["Is reference?"] = mapping["is_reference"]
                        mapping_records.append(record)
                    mapping_df = pd.DataFrame(mapping_records)
                    if df_name == "inputs":
                        inputs_mapping_df = mapping_df
                    else:
                        outputs_mapping_df = mapping_df
                else:
                    if df_name == "inputs":
                        inputs_mapping_df = pd.DataFrame()
                    else:
                        outputs_mapping_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing mapping: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    with st.expander(
        T["exp_inmap"],
        expanded=(st.session_state.get("expanded_section") == "input_mapping"),
    ):
        if inputs_mapping_df.empty:
            if saved_df_inputs.empty:
                st.info(
                    "No inputs mapping available. Open **Input Data**, select a pathway with input data, then return here."
                )
            else:
                st.info("No inputs mapping available. The mapping process may have failed or there are no matching flows in the mapping file.")
        else:
            st.dataframe(
                inputs_mapping_df,
                column_config=mapping_dataframe_column_config("input data"),
                hide_index=True,
                use_container_width=True,
            )

    with st.expander(
        T["exp_outmap"],
        expanded=(st.session_state.get("expanded_section") == "output_mapping"),
    ):
        if outputs_mapping_df.empty:
            if saved_df_outputs.empty:
                st.info(
                    "No outputs mapping available. Open **Output Data**, select a pathway with output data, then return here."
                )
            else:
                st.info("No outputs mapping available. The mapping process may have failed or there are no matching flows in the mapping file.")
        else:
            st.dataframe(
                outputs_mapping_df,
                column_config=mapping_dataframe_column_config("output data"),
                hide_index=True,
                use_container_width=True,
            )

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
        T["exp_impact"],
        expanded=(st.session_state.get("expanded_section") == "impact_results"),
    ):
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
                    with st.spinner("Please wait. Calculating impacts…"):
                        semantic_results_df = calculate_impacts(semantic_mapped, IDEMAT_SHEET, column_of_interest)
                    if not semantic_results_df.empty:
                        semantic_results_df["Type"] = "Input (Semantic Mapping)"
                        total_results_df = pd.concat([total_results_df, semantic_results_df], ignore_index=True)
                except Exception as e:
                    st.warning(f"Could not calculate impacts for semantic mapping: {str(e)}")

        if not inputs_mapping_df.empty:
            column_of_interest = "Carbon footprint (kg CO2 equiv.)"
            try:
                with st.spinner("Please wait. Calculating impacts…"):
                    inputs_results_df = calculate_impacts(inputs_mapping_df, IDEMAT_SHEET, column_of_interest, flow_direction="input")
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
                with st.spinner("Please wait. Calculating impacts…"):
                    outputs_results_df = calculate_impacts(outputs_mapping_df, IDEMAT_SHEET, column_of_interest, flow_direction="output")
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
            total_results_display_df = total_results_df.copy()
            if "Note" in total_results_display_df.columns:
                total_results_display_df = total_results_display_df[
                    ~total_results_display_df["Note"].fillna("").str.contains("Reference product", regex=False)
                ].copy()

            st.session_state["impact_results"] = total_results_display_df
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

            col_config = {"Calculated Result": st.column_config.NumberColumn("Calculated Result (kg CO2 eq.)", format="%.3f")}
            if "Note" in total_results_display_df.columns:
                col_config["Note"] = st.column_config.TextColumn("Note", help="Reason for 0 or conversion used")
            st.dataframe(
                total_results_display_df,
                hide_index=True,
                use_container_width=True,
                column_config=col_config,
            )

            if "Calculated Result" in total_results_display_df.columns:
                total_impact = float(total_results_display_df["Calculated Result"].sum())
                st.metric(T["metric_h2"], f"{total_impact:.3f} kg CO₂ eq./kg H2")
        else:
            if inputs_mapping_df.empty and outputs_mapping_df.empty:
                st.info("No input or output mapping data available")
            else:
                st.warning("No impact results calculated")

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    with st.expander(
        T["exp_viz"],
        expanded=(st.session_state.get("expanded_section") == "visualization"),
    ):
        chart_type = st.selectbox(
            T["chart_type"],
            ["Pie Chart", "Bar Chart", "Line Chart"],
            key="chart_type_selector",
            help="Choose how to display impacts by contribution category",
        )

        if st.button(
            T["generate_chart"],
            key="generate_chart_btn",
            use_container_width=True,
            help="Build the chart from the latest climate change impact results in this session",
        ):
            # Get chart data from session_state (stored in Impact Results section)
            chart_data = st.session_state.get("impact_results", pd.DataFrame())

            if chart_data is None or chart_data.empty:
                st.warning(
                    "No data available for chart generation. **Run the Climate Change Impact section first** "
                    "(select a pathway and let the impact table load), then try Generate Chart again."
                )
            elif "Calculated Result" not in chart_data.columns:
                st.error(
                    "Impact data is missing the **Calculated Result** column. Re-run the Climate Change Impact section, "
                    "then try again. If the problem persists, check that the Idemat file and mapping are loaded correctly."
                )
            else:
                try:
                    from utils.visualize_plotly import prepare_category_data

                    with st.spinner("Please wait. Generating chart…"):
                        category_impacts = prepare_category_data(chart_data)
                        total_impact = float(category_impacts.sum()) if category_impacts is not None else 0.0
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
                    st.markdown(f"## {T['category_legend']}")

                    if category_impacts is not None and len(category_impacts) > 0:
                        sorted_categories = category_impacts.sort_values(ascending=False)

                        for category, impact_value in sorted_categories.items():
                            percentage = (impact_value / total_impact * 100) if total_impact > 0 else 0.0
                            color = color_mapping.get(category, "#cccccc")

                            st.markdown(
                                f"""
                                <div style="
                                    background-color: {color};
                                    padding: 12px;
                                    border-radius: 5px;
                                    margin-bottom: 15px;
                                    border: 2px solid #000000;
                                    min-height: 80px;
                                    width: 100%;
                                    box-sizing: border-box;
                                ">
                                    <p style="margin: 0; font-weight: bold; color: #000000; font-size: 14px;">
                                        {category}
                                    </p>
                                    <p style="margin: 5px 0 0 0; color: #000000; font-size: 13px; font-weight: 600;">
                                        {impact_value:.3f} kg CO₂ eq./kg H2 ({percentage:.2f}%)
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No category impact breakdown available to display.")

                except ModuleNotFoundError as e:
                    if "plotly" in str(e).lower():
                        st.error(
                            "**Plotly is not installed.** Install it with: `pip install plotly` (or `pip install -r requirements.txt`). "
                            "Then restart the app."
                        )
                    else:
                        st.error(f"Missing dependency: {e}")
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
