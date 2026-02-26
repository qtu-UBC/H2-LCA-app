import os
import re
from typing import Optional

import pandas as pd
import pint

from config.config import LOG_DIR

def calculate_impacts(
    mapped_df: pd.DataFrame,
    idemat_datasheet: str,
    column_of_interest: str,
    flow_direction: Optional[str] = None,
) -> pd.DataFrame:
    """
    Looks up values from Idemat datasheet and multiplies with Amount column to calculate results.
    
    Args:
        mapped_df: DataFrame containing mapped flow names with 'Mapped Flow' and 'Amount' columns
        idemat_datasheet: Path to Idemat Excel file
        column_of_interest: Name of column to look up values from
        
    Returns:
        DataFrame with Mapped Flow names and calculated results
    """
    # Set up logging
    # Create calculation results log file
    log_file = os.path.join(LOG_DIR, 'calculation_results.log')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Reset log file by opening in write mode
    with open(log_file, 'w') as f:
        f.write("Calculation Results Log\n")
        f.write("=====================\n\n")
    # Initialize results DataFrame and results list
    results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category', 'Note'])
    results = []
    skipped_categories = {}
    processed_categories = {}
    
    if mapped_df.empty:
        return results_df
    
    try:
        # Read Idemat datasheet
        idemat_df = pd.read_excel(idemat_datasheet)
        
        # Create lookup series from idemat datasheet (index = Process name)
        idemat_df = idemat_df.dropna(subset=['Process'])
        idemat_df['Process'] = idemat_df['Process'].astype(str).str.strip()
        # Deduplicate by Process so indexing never returns a Series (avoids "truth value of a Series is ambiguous")
        idemat_unique = idemat_df.drop_duplicates(subset=['Process'], keep='first')
        lookup_series = pd.Series(
            idemat_unique.set_index('Process')[column_of_interest]
        )
        unit_series = pd.Series(
            idemat_unique.set_index('Process')['unit']
        )

        def _scalar(s, key):
            """Get scalar from series lookup; if duplicate index returns Series, take first."""
            v = s.get(key) if hasattr(s, 'get') else (s[key] if key in s.index else None)
            if hasattr(v, 'iloc'):
                return v.iloc[0]
            return v
        # Case-insensitive fallback: map normalized (lowercase, strip) name -> exact Idemat Process name
        # so LCI names like "argon", "carbon dioxide, liquid" match Idemat "Argon", "Carbon dioxide, liquid"
        process_names = lookup_series.index.tolist()
        idemat_lower_to_process = {}
        for p in process_names:
            key = str(p).strip().lower()
            if key and key not in idemat_lower_to_process:
                idemat_lower_to_process[key] = p

        # Import pubchempy for chemical name lookups
        import pubchempy as pcp

        # Known LCI name -> Idemat Process (same compound, different name)
        lci_to_idemat_synonyms = {
            "dinitrogen monoxide": "Nitrous oxide",
            "nitrogen oxides": "Nitric acid",
            "nitrous oxide": "Nitrous oxide",
        }

        def resolve_idemat_process(mapped_flow):
            """Return Idemat Process name for this flow, or None if not found.
            Tries: exact match, case-insensitive, synonyms, first-segment match, contains/startswith, water fallback.
            """
            if not mapped_flow or (isinstance(mapped_flow, float) and pd.isna(mapped_flow)):
                return None
            s = str(mapped_flow).strip()
            if s in lookup_series.index:
                return s
            key = s.lower()
            if key in idemat_lower_to_process:
                return idemat_lower_to_process[key]
            # Synonym lookup (e.g. Dinitrogen monoxide -> Nitrous oxide)
            if key in lci_to_idemat_synonyms:
                syn = lci_to_idemat_synonyms[key]
                if syn in lookup_series.index:
                    return syn
            # First-segment match: "Carbon monoxide, fossil" -> "Carbon monoxide"
            key_first = key.split(",")[0].strip()
            for proc in process_names:
                proc_lower = str(proc).strip().lower()
                proc_first = proc_lower.split(",")[0].strip()
                if key_first == proc_first or (key.startswith(proc_first) or proc_first.startswith(key_first)):
                    return proc
            # Contains or startswith (e.g. "nitrogen" -> "Nitrogen, liquid")
            for proc in process_names:
                proc_lower = str(proc).strip().lower()
                if key in proc_lower or proc_lower.startswith(key):
                    return proc
            # Water flows: "Water, CA" etc. -> prefer water supply processes
            if key.startswith("water"):
                for wp in ("Deionized water production", "drinking water europe", "industrial reverse osmosis water europe", "Waste water treatment"):
                    if wp in lookup_series.index:
                        return wp
            return None

        # Initialize Pint unit registry
        ureg = pint.UnitRegistry()

        def _normalize_pint_unit(unit_val):
            """Normalize CSV/Idemat unit spellings to Pint-compatible units."""
            if unit_val is None or pd.isna(unit_val):
                return None
            u = str(unit_val).strip()
            if not u:
                return None
            unit_map = {
                "m3": "meter**3",
                "m^3": "meter**3",
                "m³": "meter**3",
                "dm3": "decimeter**3",
                "dm^3": "decimeter**3",
                "cm3": "centimeter**3",
                "cm^3": "centimeter**3",
                "l": "liter",
                "L": "liter",
                "kWh": "kilowatt_hour",
            }
            return unit_map.get(u, u)

        def _extract_original_flow_name(row_obj, fallback_name):
            raw = row_obj.get('Original Flow', fallback_name)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                raw = fallback_name
            s = str(raw).strip()
            if "_" in s:
                head, tail = s.rsplit("_", 1)
                if tail.isdigit():
                    return head
            return s

        def _is_elementary_category(cat):
            if cat is None or (isinstance(cat, float) and pd.isna(cat)):
                return False
            return "elementary flows" in str(cat).strip().lower()

        def _is_likely_ghg(flow_name):
            if not flow_name:
                return False
            s = str(flow_name).strip().lower()
            ghg_patterns = [
                r"\bcarbon dioxide\b",
                r"\bmethane\b",
                r"\bnitrous oxide\b",
                r"\bdinitrogen monoxide\b",
                r"\bsulfur hexafluoride\b",
                r"\bnitrogen trifluoride\b",
                r"\bhfc[-\s]?\d+\b",
                r"\bpfc[-\s]?\d+\b",
                r"\bperfluoro",
                r"\bhydrofluoro",
                r"\bcf4\b",
                r"\bc2f6\b",
                r"\bc3f8\b",
                r"\bc4f10\b",
                r"\bcfc[-\s]?\d+\b",
                r"\bsf6\b",
                r"\bnf3\b",
            ]
            return any(re.search(p, s) for p in ghg_patterns)
        
        # Calculate results by multiplying Amount with looked up values
        for _, row in mapped_df.iterrows():
            mapped_flow = row.get('Mapped Flow', 'Unknown')
            contribution_cat = row.get('Contribution Category') or row.get('Category', 'Unknown')
            try:
                mapped_flow = row['Mapped Flow']
            
                # Get Contribution Category for tracking
                contribution_cat = None
                if 'Contribution Category' in row:
                    contribution_cat = row['Contribution Category']
                if not contribution_cat or pd.isna(contribution_cat) or str(contribution_cat).strip() == "":
                    contribution_cat = row.get('Category', 'Unknown')
            
                # Helper to add a result row (used for both successful and skipped flows so Impact Results shows all flows)
                def add_result_row(mapped_flow_name, calculated_val, contrib_cat, cat, note=""):
                    result_record = {
                        'Mapped Flow': mapped_flow_name,
                        'Calculated Result': calculated_val,
                        'Category': cat,
                        'Note': note,
                    }
                    result_record['Contribution Category'] = (contrib_cat if contrib_cat and str(contrib_cat).strip() else cat) if cat else (contrib_cat or 'Unknown')
                    results.append(result_record)

                is_reference = False
                if 'Is reference?' in row and pd.notna(row['Is reference?']):
                    ref_val = str(row['Is reference?']).strip().lower()
                    is_reference = ref_val in {"x", "true", "1", "yes", "y"}

                category_value = row.get('Category', '')
                original_flow_name = _extract_original_flow_name(row, mapped_flow)
                is_elementary = _is_elementary_category(category_value)

                # Reference product and non-elementary outputs (incl. co-products) are not
                # characterized as climate-change contributors in this app view.
                if is_reference:
                    add_result_row(mapped_flow, 0.0, contribution_cat, category_value, note="Reference product")
                    continue
                if (flow_direction or "").lower() == "output" and not is_elementary:
                    add_result_row(mapped_flow, 0.0, contribution_cat, category_value, note="Output non-elementary flow")
                    continue

                # For elementary flows, only GHGs should contribute to climate change.
                if is_elementary and not _is_likely_ghg(original_flow_name):
                    add_result_row(mapped_flow, 0.0, contribution_cat, category_value, note="Elementary non-GHG flow")
                    continue
            
                # Resolve to Idemat Process name: exact match, then case-insensitive, then PubChem synonyms
                idemat_process = resolve_idemat_process(mapped_flow)
                if idemat_process is None:
                    try:
                        compounds = pcp.get_compounds(mapped_flow, 'name')
                        if compounds:
                            for synonym in compounds[0].synonyms:
                                idemat_process = resolve_idemat_process(synonym)
                                if idemat_process is not None:
                                    break
                    except Exception:
                        pass
                if idemat_process is not None:
                    # Get units (strip spaces; Idemat Excel may have e.g. 'kg ')
                    source_unit = row.get('Unit', None)  # caution, capital U
                    if source_unit is not None and not pd.isna(source_unit):
                        source_unit = str(source_unit).strip() or None
                    dest_unit = None
                    if idemat_process in unit_series.index:
                        du = _scalar(unit_series, idemat_process)
                        if du is not None and not pd.isna(du):
                            dest_unit = str(du).strip() or None
                
                    # Check if units are valid
                    if pd.isna(dest_unit) or dest_unit is None or str(dest_unit).strip() == "":
                        # If dest_unit is missing, try to use source_unit or skip conversion
                        if source_unit and pd.notna(source_unit) and str(source_unit).strip() != "":
                            dest_unit = source_unit
                        else:
                            # No valid units, still show in results with 0
                            skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                            add_result_row(idemat_process, 0.0, contribution_cat, row['Category'], note="Missing unit in Idemat")
                            continue
                
                    if not source_unit or pd.isna(source_unit) or str(source_unit).strip() == "":
                        # No source unit, still show in results with 0
                        skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                        add_result_row(idemat_process, 0.0, contribution_cat, row['Category'], note="Missing source unit")
                        continue
                
                    try:
                        # Convert amount to destination unit
                        source_unit_s = _normalize_pint_unit(source_unit)
                        dest_unit_s = _normalize_pint_unit(dest_unit)
                        source_quantity = row['Amount'] * ureg(source_unit_s)
                        dest_quantity = source_quantity.to(dest_unit_s)
                        converted_amount = dest_quantity.magnitude
                    
                        # Get CF value (treat NaN as 0 so result is 0, not NaN)
                        cf_value = _scalar(lookup_series, idemat_process)
                        if cf_value is None or pd.isna(cf_value):
                            cf_value = 0.0
                    
                        # Calculate result with converted amount
                        calculated_result = converted_amount * cf_value
                        result_record = {
                            'Mapped Flow': idemat_process,
                            'Calculated Result': calculated_result,
                            'Category': row['Category'],
                            'Note': '',
                        }
                    
                        # Add Contribution Category from input data, preserving user's custom values
                        if 'Contribution Category' in row:
                            result_contribution_cat = row['Contribution Category']
                            if pd.notna(result_contribution_cat) and str(result_contribution_cat).strip() != "":
                                result_record['Contribution Category'] = str(result_contribution_cat).strip()
                            else:
                                result_record['Contribution Category'] = row['Category']
                        else:
                            result_record['Contribution Category'] = contribution_cat if contribution_cat else row['Category']
                    
                        final_cat = result_record.get('Contribution Category', 'Unknown')
                        processed_categories[final_cat] = processed_categories.get(final_cat, 0) + 1
                        results.append(result_record)
                    except (pint.errors.DimensionalityError, AttributeError, ValueError, TypeError) as e:
                        # Try water volume<->mass fallback: 1 L water ≈ 1 kg
                        converted_amount = None
                        su = str(source_unit).strip().lower() if source_unit else ""
                        du = str(dest_unit).strip().lower() if dest_unit else ""
                        vol_units = ('l', 'liter', 'litre', 'm3', 'dm3')
                        mass_units = ('kg', 'g')
                        amount = row['Amount']
                        if su in vol_units and du in mass_units:
                            if su in ('m3',):
                                converted_amount = amount * 1000.0
                            else:
                                converted_amount = amount
                        elif su in mass_units and du in vol_units:
                            if du == 'm3':
                                converted_amount = amount / 1000.0
                            else:
                                converted_amount = amount
                        if converted_amount is not None:
                            cf_value = _scalar(lookup_series, idemat_process)
                            if cf_value is None or pd.isna(cf_value) or cf_value == 0:
                                water_processes_kg = [
                                    "Deionized water production",
                                    "drinking water europe",
                                    "industrial reverse osmosis water europe",
                                ]
                                for wp in water_processes_kg:
                                    if wp in lookup_series.index:
                                        cv = _scalar(lookup_series, wp)
                                        if cv is not None and pd.notna(cv) and cv != 0:
                                            idemat_process = wp
                                            cf_value = float(cv)
                                            break
                                else:
                                    cf_value = 0.0
                            else:
                                cf_value = float(cf_value)
                            calculated_result = converted_amount * cf_value
                            result_record = {
                                'Mapped Flow': idemat_process,
                                'Calculated Result': calculated_result,
                                'Category': row['Category'],
                                'Note': 'Volume–mass conversion (water, 1 L≈1 kg)',
                            }
                            if 'Contribution Category' in row and pd.notna(row['Contribution Category']) and str(row['Contribution Category']).strip():
                                result_record['Contribution Category'] = str(row['Contribution Category']).strip()
                            else:
                                result_record['Contribution Category'] = contribution_cat if contribution_cat else row['Category']
                            results.append(result_record)
                            continue
                        skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                        with open(log_file, 'a') as f:
                            f.write(f"Skipped {idemat_process}: Unit conversion error - {str(e)}\n")
                        add_result_row(idemat_process, 0.0, contribution_cat, row['Category'], note="Unit conversion error")
                        continue
                else:
                    # Flow not found in Idemat lookup: still show in results with 0 so Impact Results lists all flows
                    skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                    add_result_row(mapped_flow, 0.0, contribution_cat, row['Category'], note="No match in Idemat")
            except Exception as e:
                # Ensure this flow still appears in the table (no missing flows)
                results.append({
                    'Mapped Flow': mapped_flow,
                    'Calculated Result': 0.0,
                    'Category': row.get('Category', ''),
                    'Contribution Category': contribution_cat,
                    'Note': 'Calculation error',
                })
                with open(log_file, 'a') as f:
                    f.write(f"Error for flow {mapped_flow}: {e}\n")

        # Create results DataFrame from results list
        if len(results) > 0:
            results_df = pd.DataFrame(results)
        else:
            # If no results, return empty DataFrame with correct columns
            results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category', 'Note'])
            
    except Exception as e:
        print(f"Error calculating values: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty DataFrame with correct columns on error
        results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category', 'Note'])
        
    return results_df


def compute_pathway_impact(
    inputs_df: pd.DataFrame,
    outputs_df: pd.DataFrame,
    matching_source_files: list,
    mapping_file: str,
    idemat_sheet: str,
    column_of_interest: str = "Carbon footprint (kg CO2 equiv.)",
) -> float:
    """
    Single source of truth for pathway climate change impact (GWP).
    Uses same data (CSV) and logic as Climate Change Impact: map_flows + IDEMAT.
    Returns total impact (kg CO2 eq.) for the pathway so all reporting stays consistent.

    Args:
        inputs_df: Full inputs dataframe (must have Source_File column).
        outputs_df: Full outputs dataframe (must have Source_File column).
        matching_source_files: List of Source_File values that identify this pathway.
        mapping_file: Path to mapping CSV.
        idemat_sheet: Path to Idemat Excel.
        column_of_interest: Column name in Idemat for CF (default GWP).

    Returns:
        Total climate change impact (sum of Calculated Result) for the pathway.
    """
    from utils.mapper import map_flows

    total = 0.0
    if not matching_source_files:
        return total

    if "Source_File" not in inputs_df.columns and "Source_File" not in outputs_df.columns:
        return total

    pathway_inputs = (
        inputs_df[inputs_df["Source_File"].isin(matching_source_files)]
        if "Source_File" in inputs_df.columns
        else pd.DataFrame()
    )
    pathway_outputs = (
        outputs_df[outputs_df["Source_File"].isin(matching_source_files)]
        if "Source_File" in outputs_df.columns
        else pd.DataFrame()
    )

    for _name, df in [("inputs", pathway_inputs), ("outputs", pathway_outputs)]:
        if df.empty:
            continue
        try:
            flow_mappings = map_flows(df, mapping_file)
            mapping_records = []
            for _key, m in flow_mappings.items():
                rec = {
                    "Mapped Flow": m.get("mapped_flow", ""),
                    "Amount": m.get("amount"),
                    "Unit": m.get("unit", ""),
                    "Category": m.get("category", ""),
                    "Contribution Category": m.get("contribution_category", m.get("category", "")),
                }
                if m.get("is_reference") is not None:
                    rec["Is reference?"] = m["is_reference"]
                mapping_records.append(rec)
            mapping_df = pd.DataFrame(mapping_records)
            if mapping_df.empty:
                continue
            results = calculate_impacts(
                mapping_df,
                idemat_sheet,
                column_of_interest,
                flow_direction=("input" if _name == "inputs" else "output"),
            )
            if not results.empty and "Calculated Result" in results.columns:
                # Exclude reference product from pathway total (it's the functional unit, not an input)
                mask = ~results["Note"].fillna("").str.contains("Reference product", regex=False)
                total += float(results.loc[mask, "Calculated Result"].sum())
        except Exception:
            continue

    return total
