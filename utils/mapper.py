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
    
    # Create mapping series using both 'Unique Flow Name' and 'Location'
    # Create a composite key for more accurate mapping
    dest_df['composite_key'] = dest_df['Unique Flow Name'] + '|' + dest_df['Location'].fillna('')
    mapping_series = pd.Series(
        dest_df.set_index('composite_key')['Most Similar Process']
    )
    
    # Also keep the original mapping for fallback
    simple_mapping_series = pd.Series(
        dest_df.set_index('Unique Flow Name')['Most Similar Process']
    )
    
    # Iterate through rows and create mappings
    for idx, row in mapped_df.iterrows():
        try:
            # Try mapping Provider first
            flow = row['Provider']
            mapped_flow = None
            
            # Check if Provider exists and has a location for composite mapping
            if not pd.isna(flow) and 'Location' in row:
                composite_key = flow + '|' + (row['Location'] if not pd.isna(row['Location']) else '')
                if composite_key in mapping_series:
                    mapped_flow = mapping_series[composite_key]
            
            # If no composite mapping found, try simple Provider mapping
            if mapped_flow is None and not pd.isna(flow) and flow in simple_mapping_series:
                mapped_flow = simple_mapping_series[flow]
            
            # If no Provider mapping found, try mapping Flow
            if mapped_flow is None:
                flow = row['Flow']
                # Try composite mapping for Flow
                if 'Location' in row:
                    composite_key = flow + '|' + (row['Location'] if not pd.isna(row['Location']) else '')
                    if composite_key in mapping_series:
                        mapped_flow = mapping_series[composite_key]
                
                # Fallback to simple mapping
                if mapped_flow is None and flow in simple_mapping_series:
                    mapped_flow = simple_mapping_series[flow]
                elif mapped_flow is None:
                    mapped_flow = flow
            
            mapping_dict = {
                'mapped_flow': mapped_flow,
                'amount': row['Amount'],
                'unit': row['Unit'],
                'category': row['Category'],
                'original_flow': flow
            }
            
            # Add 'Is reference?' if it exists in source_df
            if 'Is reference?' in row:
                mapping_dict['is_reference'] = row['Is reference?']
                
            # Add 'Location' if it exists in source_df
            if 'Location' in row and not pd.isna(row['Location']):
                mapping_dict['location'] = row['Location']
                
            flow_mappings[f"{flow}_{idx}"] = mapping_dict
            
            # Update the flow in mapped_df
            mapped_df.at[idx, 'Flow'] = mapped_flow
            
        except Exception as e:
            print(f"Error mapping flow '{row['Flow']}': {str(e)}")
            # # Use original flow as fallback
            # flow = row['Flow']
            # flow_mappings[f"{flow}_{idx}"] = {
            #     'mapped_flow': flow,
            #     'amount': row['Amount'],
            #     'unit': row['Unit'],
            #     'original_flow': flow  
            # }
            # mapped_df.at[idx, 'Flow'] = flow
            
    return flow_mappings
