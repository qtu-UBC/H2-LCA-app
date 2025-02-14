import pandas as pd
import pint

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
    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=['Mapped Flow', 'Calculated Result'])
    
    try:
        # Read Idemat datasheet
        idemat_df = pd.read_excel(idemat_datasheet)
        # Print dimensions of idemat datasheet
        # print(f"Idemat datasheet dimensions: {idemat_df.shape}")
        
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
        results = []
        for _, row in mapped_df.iterrows():
            mapped_flow = row['Mapped Flow']
            
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
                                print(f"Found matching synonym: {synonym} for mapped flow: {mapped_flow}")
                                flow_to_use = synonym
                                break
                        
                        if not flow_to_use:
                            flow_to_use = mapped_flow
                else:
                    flow_to_use = mapped_flow
            except Exception as e:
                print(f"Error getting synonyms for {mapped_flow}: {str(e)}")
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
                    print(f"Converting {row['Amount']} {source_unit} to {converted_amount} {dest_unit}")
                    
                    # Calculate result with converted amount
                    calculated_result = converted_amount * lookup_series[mapped_flow]
                    results.append({
                        'Mapped Flow': mapped_flow,
                        'Calculated Result': calculated_result
                    })
                except pint.errors.DimensionalityError:
                    print(f"Cannot convert between units {source_unit} and {dest_unit}")
                    continue
                
        results_df = pd.DataFrame(results)
            
    except Exception as e:
        print(f"Error calculating values: {str(e)}")
        
    return results_df
