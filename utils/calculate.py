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
        print(f"Idemat datasheet dimensions: {idemat_df.shape}")
        
        # Create lookup series from idemat datasheet
        lookup_series = pd.Series(
            idemat_df.set_index('Process')[column_of_interest]
        )
        
        # Create lookup series for units from idemat datasheet
        unit_series = pd.Series(
            idemat_df.set_index('Process')['unit']
        )
        
        # Initialize Pint unit registry
        ureg = pint.UnitRegistry()
        
        # Calculate results by multiplying Amount with looked up values
        results = []
        for _, row in mapped_df.iterrows():
            mapped_flow = row['Mapped Flow']
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
