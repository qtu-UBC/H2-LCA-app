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
        return flow_mappings

    if dest_df.empty:
        return flow_mappings

    # Create a copy of source_df
    mapped_df = source_df.copy()

    # Create mapping series using both 'Unique Flow Name' and 'Location'
    # Create a composite key for more accurate mapping
    dest_df['composite_key'] = dest_df['Unique Flow Name'].fillna('').astype(str) + '|' + dest_df['Location'].fillna('')
    mapping_series = pd.Series(
        dest_df.set_index('composite_key')['Most Similar Process']
    )
    
    # Fallback: one row per Unique Flow Name (avoid duplicate index so lookup returns a scalar)
    simple_mapping_series = (
        dest_df.drop_duplicates(subset=['Unique Flow Name'], keep='first')
        .set_index('Unique Flow Name')['Most Similar Process']
    )
    
    # Iterate through rows and create mappings
    for idx, row in mapped_df.iterrows():
        try:
            # Try mapping Provider first (use .get so missing column doesn't raise)
            flow = row.get('Provider', row.get('Flow', ''))
            if pd.isna(flow) or not str(flow).strip():
                flow = row.get('Flow', '')
            flow = str(flow).strip() if pd.notna(flow) and str(flow).strip() else ''
            if not flow:
                flow = str(row.get('Flow', '')).strip()
            mapped_flow = None

            # Check if Provider/flow exists and has a location for composite mapping
            if flow and 'Location' in row:
                loc_val = row['Location']
                loc_str = str(loc_val).strip() if pd.notna(loc_val) and str(loc_val).strip() else ''
                composite_key = flow + '|' + loc_str
                if composite_key in mapping_series.index:
                    mapped_flow = mapping_series[composite_key]
            
            # If no composite mapping found, try simple Provider mapping
            if mapped_flow is None and flow and flow in simple_mapping_series.index:
                mapped_flow = simple_mapping_series[flow]
            
            # If no Provider mapping found, try mapping Flow (flow already set from Provider or Flow)
            if mapped_flow is None and flow and 'Location' in row:
                loc_val = row['Location']
                loc_str = str(loc_val).strip() if pd.notna(loc_val) and str(loc_val).strip() else ''
                composite_key = flow + '|' + loc_str
                if composite_key in mapping_series.index:
                    mapped_flow = mapping_series[composite_key]
            
            if mapped_flow is None and flow and flow in simple_mapping_series.index:
                mapped_flow = simple_mapping_series[flow]
            elif mapped_flow is None:
                mapped_flow = flow

            flow_str = flow
            mapping_dict = {
                'mapped_flow': mapped_flow,
                'amount': row.get('Amount'),
                'unit': row.get('Unit'),
                'category': row.get('Category', ''),
                'original_flow': flow_str
            }
            
            # Add 'Contribution Category' from input data, preserving user's custom values
            # Use whatever Contribution Category is in the input data, or fall back to Category
            if 'Contribution Category' in row:
                contribution_cat = row['Contribution Category']
                # Use the Contribution Category from input data if it's not empty/NaN
                if pd.notna(contribution_cat) and str(contribution_cat).strip() != "":
                    mapping_dict['contribution_category'] = str(contribution_cat).strip()
                else:
                    # Empty or NaN, use Category as default
                    mapping_dict['contribution_category'] = row['Category']
            else:
                # No Contribution Category column, use Category
                mapping_dict['contribution_category'] = row['Category']
            
            # Add 'Is reference?' if it exists in source_df
            if 'Is reference?' in row:
                mapping_dict['is_reference'] = row['Is reference?']
                
            # Add 'Location' if it exists in source_df
            if 'Location' in row and not pd.isna(row['Location']):
                mapping_dict['location'] = row['Location']
                
            flow_mappings[f"{flow_str}_{idx}"] = mapping_dict
            
            # Update the flow in mapped_df
            if 'Flow' in mapped_df.columns:
                mapped_df.at[idx, 'Flow'] = mapped_flow
            
        except Exception as e:
            print(f"Error mapping flow '{row.get('Flow', row.get('Provider', idx))}': {str(e)}")
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
