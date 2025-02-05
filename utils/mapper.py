import pandas as pd

def map_flows(source_df: pd.DataFrame, destination_file: str) -> dict:
    """
    Maps source flows to a destination file using semantic similarity.
    
    Args:
        source_df: DataFrame containing flows to map from
        destination_file: Path to file containing destination flows to map to
        
    Returns:
        Dictionary mapping source flows to destination flows
    """
    # Initialize mapping dictionary
    flow_mappings = {}
    
    # Read destination file
    try:
        dest_df = pd.read_csv(destination_file)
    except Exception as e:
        print(f"Error reading destination file {destination_file}: {str(e)}")
        return flow_mappings
        
    # Create a copy of source_df
    mapped_df = source_df.copy()
    
    # Create mapping series using merge
    mapping_series = pd.Series(
        dest_df.set_index('Unique Flow Name')['Most Similar Process']
    )
    
    # Map flows using vectorized operations
    flow_mappings = mapping_series.reindex(mapped_df['Flow']).to_dict()
    
    # Update mapped_df flows in one operation
    mapped_df['Flow'] = mapped_df['Flow'].map(flow_mappings)
            
    return flow_mappings
