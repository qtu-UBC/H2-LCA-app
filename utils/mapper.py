import pandas as pd

def map_flows(source_df: pd.DataFrame, destination_file: str) -> dict:
    """
    Maps source flows to a destination file using semantic similarity.
    Preserves the original flow quantities in the mapping.
    Handles multiple occurrences of the same flow by retaining each instance.
    
    Args:
        source_df: DataFrame containing flows to map from
        destination_file: Path to file containing destination flows to map to
        
    Returns:
        Dictionary mapping source flows to destination flows with quantities
    """
    # Initialize mapping dictionary
    flow_mappings = {}
    
    # Read destination file
    try:
        dest_df = pd.read_csv(destination_file)
    except Exception as e:
        print(f"Error reading destination file {destination_file}: {str(e)}")
        
    # Create a copy of source_df
    mapped_df = source_df.copy()
    
    # Create mapping series using merge
    mapping_series = pd.Series(
        dest_df.set_index('Unique Flow Name')['Most Similar Process']
    )
    
    # Iterate through rows and create mappings
    for idx, row in mapped_df.iterrows():
        flow = row['Flow']
        mapped_flow = mapping_series.get(flow, None)
        
        # If no mapping found, use original flow
        if pd.isna(mapped_flow):
            mapped_flow = flow
        
        flow_mappings[f"{flow}_{idx}"] = {
            'mapped_flow': mapped_flow,
            'amount': row['Amount'],
            'unit': row['Unit'],
            'original_flow': flow
        }
        
        # Update the flow in mapped_df
        mapped_df.at[idx, 'Flow'] = mapped_flow
            
    return flow_mappings

