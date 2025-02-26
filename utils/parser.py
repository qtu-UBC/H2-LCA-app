


"""
======
Import libraries
======
"""
import os
import pandas as pd

"""
======
Function to parse OpenLCA export Excel files
======
"""
def openlca_export_parser(export_folder_path: str, tab_of_interest: list) -> dict:
    """
    Parse OpenLCA export Excel files and extract data from specified tabs into a dictionary.
    Also extracts unique flow values, provider names, and locations for Inputs and Outputs tabs.
    
    Args:
        export_folder_path: Path to folder containing OpenLCA export Excel files
        tab_of_interest: List of tab names to parse
        
    Returns:
        Dictionary containing parsed data from specified tabs, unique flows, providers and locations
    """
    
    results = {}
    unique_flows = {'Inputs': set(), 'Outputs': set()}
    unique_providers = {'Inputs': set(), 'Outputs': set()}
    unique_locations = {'Inputs': set(), 'Outputs': set()}
    
    # Loop through all Excel files in the export folder
    for file in os.listdir(export_folder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(export_folder_path, file)
            
            # Read each tab of interest from the Excel file
            for tab in tab_of_interest:
                try:
                    # Read the Excel sheet into a DataFrame
                    df = pd.read_excel(file_path, sheet_name=tab)
                    
                    # Remove empty rows
                    df = df.dropna(how='all')
                    
                    # If tab is Inputs, Outputs, Providers or Locations, collect unique flows, providers and locations
                    if tab in ['Inputs', 'Outputs','Providers','Locations']:
                        if 'Flow' in df.columns:
                            unique_flows[tab].update(df['Flow'].unique())
                        if 'Provider' in df.columns:
                            unique_providers[tab].update(df['Provider'].dropna().unique())
                        if 'Location' in df.columns:
                            unique_locations[tab].update(df['Location'].dropna().unique())
                    
                    # Convert DataFrame to dictionary
                    tab_dict = df.to_dict('records')
                    
                    # Store in results dictionary using file name and tab as keys
                    file_key = os.path.splitext(file)[0]
                    if file_key not in results:
                        results[file_key] = {}
                    results[file_key][tab] = tab_dict
                    
                except Exception as e:
                    print(f"Error processing tab {tab} in file {file}: {str(e)}")
                    
    # Convert parsed data into DataFrames organized by tab
    merged_dfs = {}
    for tab in tab_of_interest:
        tab_dfs = []
        for file_key, file_data in results.items():
            if tab in file_data:
                df = pd.DataFrame(file_data[tab])
                df['Source_File'] = file_key  # Add source file column
                tab_dfs.append(df)
        
        if tab_dfs:
            # Merge all DataFrames for this tab
            merged_dfs[tab] = pd.concat(tab_dfs, ignore_index=True)
    
    # Add merged DataFrames, unique flows, providers and locations to results
    results['merged_data'] = merged_dfs
    results['unique_flows'] = {k: list(v) for k, v in unique_flows.items() if k in tab_of_interest}
    results['unique_providers'] = {k: list(v) for k, v in unique_providers.items() if k in tab_of_interest}
    results['unique_locations'] = {k: list(v) for k, v in unique_locations.items() if k in tab_of_interest}
    
    return results

if __name__ == "__main__":
    
    from config.config import H2_LCI_FOLDER, OUTPUT_DIR

    # Define tabs to parse
    tabs_to_parse = ["General information", "Inputs", "Outputs","Providers","Locations"]
    
    # Parse the OpenLCA export files
    parsed_data = openlca_export_parser(H2_LCI_FOLDER, tabs_to_parse)

    # Export merged data for each tab to separate CSV files
    merged_data = parsed_data['merged_data']
    for tab_name, tab_df in merged_data.items():
        # Create sanitized filename from tab name
        filename = tab_name.lower().replace(" ", "_") + ".csv"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Export to CSV
        tab_df.to_csv(output_path, index=False)
        print(f"\nExported {tab_name} data to: {output_path}")

    # Export unique flows, providers and locations data to CSV
    unique_flows = parsed_data['unique_flows']
    unique_providers = parsed_data['unique_providers']
    unique_locations = parsed_data['unique_locations']
    
    unique_data_df = pd.DataFrame()
    for tab_name in ['Inputs', 'Outputs','Providers','Locations']:
        if tab_name in unique_flows:
            unique_data_df[f'{tab_name}_Flows'] = pd.Series(unique_flows[tab_name])
        if tab_name in unique_providers:
            unique_data_df[f'{tab_name}_Providers'] = pd.Series(unique_providers[tab_name])
        if tab_name in unique_locations:
            unique_data_df[f'{tab_name}_Locations'] = pd.Series(unique_locations[tab_name])
    
    unique_data_path = os.path.join(OUTPUT_DIR, "unique_flows_and_providers.csv")
    unique_data_df.to_csv(unique_data_path, index=False)
    print(f"\nExported unique flows, providers and locations data to: {unique_data_path}")
    
    # Print results summary
    print("\nParsed Data Summary:")
    for file_name, file_data in parsed_data.items():
        print(f"\nFile: {file_name}")
        for tab_name, tab_data in file_data.items():
            print(f"  Tab: {tab_name}")
            print(f"  Number of records: {len(tab_data)}")
