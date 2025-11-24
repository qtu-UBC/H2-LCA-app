import pandas as pd
import pint
import os
from config.config import LOG_DIR

def calculate_impacts(mapped_df: pd.DataFrame, idemat_datasheet: str, column_of_interest: str) -> pd.DataFrame:
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
    results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category'])
    results = []
    skipped_categories = {}
    processed_categories = {}
    
    if mapped_df.empty:
        return results_df
    
    try:
        # Read Idemat datasheet
        idemat_df = pd.read_excel(idemat_datasheet)
        
        # Create lookup series from idemat datasheet
        lookup_series = pd.Series(
            idemat_df.set_index('Process')[column_of_interest]
        )
        
        # Create lookup series for units from idemat datasheet
        unit_series = pd.Series(
            idemat_df.set_index('Process')['unit']
        )
        # Import pubchempy for chemical name lookups
        import pubchempy as pcp
        
        # Initialize Pint unit registry
        ureg = pint.UnitRegistry()
        
        # Calculate results by multiplying Amount with looked up values
        for _, row in mapped_df.iterrows():
            mapped_flow = row['Mapped Flow']
            
            # Get Contribution Category for tracking
            contribution_cat = None
            if 'Contribution Category' in row:
                contribution_cat = row['Contribution Category']
            if not contribution_cat or pd.isna(contribution_cat) or str(contribution_cat).strip() == "":
                contribution_cat = row.get('Category', 'Unknown')
            
            # Check if flow is reference flow
            if 'Is reference?' in row and pd.notna(row['Is reference?']):
                with open(log_file, 'a') as f:
                    f.write(f"Reference product: {mapped_flow}, calculation skipped\n")
                skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                continue
            
            # Get synonyms for mapped flow
            try:
                compounds = pcp.get_compounds(mapped_flow, 'name')
                if compounds:
                    # Check original mapped flow against lookup
                    if mapped_flow in lookup_series.index:
                        flow_to_use = mapped_flow
                    else:
                        # Try each synonym until we find a match
                        flow_to_use = None
                        for synonym in compounds[0].synonyms:
                            if synonym in lookup_series.index:
                                flow_to_use = synonym
                                break
                        
                        if not flow_to_use:
                            flow_to_use = mapped_flow
                else:
                    flow_to_use = mapped_flow
            except Exception as e:
                flow_to_use = mapped_flow
                
            mapped_flow = flow_to_use
            if mapped_flow in lookup_series.index:
                # Get units
                source_unit = row['Unit'] # caution, capital U
                dest_unit = unit_series[mapped_flow]
                
                try:
                    # Convert amount to destination unit
                    source_quantity = row['Amount'] * ureg(source_unit)
                    dest_quantity = source_quantity.to(dest_unit)
                    converted_amount = dest_quantity.magnitude
                    
                    # Calculate result with converted amount
                    calculated_result = converted_amount * lookup_series[mapped_flow]
                    result_record = {
                        'Mapped Flow': mapped_flow,
                        'Calculated Result': calculated_result,
                        'Category': row['Category']
                    }
                    
                    # Add Contribution Category if available, else use Category as default
                    # Use the contribution_cat we already extracted, or get it from row if needed
                    result_contribution_cat = contribution_cat
                    if 'Contribution Category' in row:
                        result_contribution_cat = row['Contribution Category']
                        # Check if it's a valid user-selected category (not empty, not NaN)
                        if pd.notna(result_contribution_cat) and str(result_contribution_cat).strip() != "":
                            # Verify it's one of the valid dropdown options
                            valid_categories = ["Feedstock", "Materials", "Energy", "Waste", "Direct Emissions"]
                            if str(result_contribution_cat).strip() in valid_categories:
                                result_record['Contribution Category'] = str(result_contribution_cat).strip()
                            else:
                                # Not a valid user selection, use Category
                                result_record['Contribution Category'] = row['Category']
                        else:
                            # Empty or NaN, use Category as default
                            result_record['Contribution Category'] = row['Category']
                    else:
                        # No Contribution Category column, use Category
                        result_record['Contribution Category'] = row['Category']
                    
                    # Track which categories were successfully processed
                    final_cat = result_record.get('Contribution Category', 'Unknown')
                    processed_categories[final_cat] = processed_categories.get(final_cat, 0) + 1
                    
                    results.append(result_record)
                except pint.errors.DimensionalityError:
                    skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                    continue
            else:
                # Flow not found in lookup_series
                skipped_categories[contribution_cat] = skipped_categories.get(contribution_cat, 0) + 1
                
        # Create results DataFrame from results list
        if len(results) > 0:
            results_df = pd.DataFrame(results)
        else:
            # If no results, return empty DataFrame with correct columns
            results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category'])
            
    except Exception as e:
        print(f"Error calculating values: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty DataFrame with correct columns on error
        results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result', 'Category', 'Contribution Category'])
        
    return results_df
