import pandas as pd
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
        
        # Create lookup series from idemat datasheet
        lookup_series = pd.Series(
            idemat_df.set_index('Process')[column_of_interest]
        )
        
        # Calculate results by multiplying Amount with looked up values
        results = []
        for _, row in mapped_df.iterrows():
            mapped_flow = row['Mapped Flow']
            if mapped_flow in lookup_series.index:
                calculated_result = row['Amount'] * lookup_series[mapped_flow]
                results.append({
                    'Mapped Flow': mapped_flow,
                    'Calculated Result': calculated_result
                })
                
        results_df = pd.DataFrame(results)
            
    except Exception as e:
        print(f"Error calculating values: {str(e)}")
        
    return results_df
