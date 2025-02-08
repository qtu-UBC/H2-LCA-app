


"""
======
Import libraries
======
"""
import os
import pandas as pd
from config.config import H2_LCI_FOLDER, OUTPUT_DIR

"""
======
Function to parse OpenLCA export Excel files
======
"""
def openlca_export_parser(export_folder_path: str, tab_of_interest: list) -> dict:
    """
    Parse OpenLCA export Excel files and extract data from specified tabs into a dictionary.
    Also extracts unique flow values for Inputs and Outputs tabs.
    
    Args:
        export_folder_path: Path to folder containing OpenLCA export Excel files
        tab_of_interest: List of tab names to parse
        
    Returns:
        Dictionary containing parsed data from specified tabs and unique flows
    """
    
    results = {}
    unique_flows = {'Inputs': set(), 'Outputs': set()}
    
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
                    
                    # If tab is Inputs or Outputs, collect unique flows
                    if tab in ['Inputs', 'Outputs'] and 'Flow' in df.columns:
                        unique_flows[tab].update(df['Flow'].unique())
                    
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
    
    # Add merged DataFrames and unique flows to results
    results['merged_data'] = merged_dfs
    results['unique_flows'] = {k: list(v) for k, v in unique_flows.items() if k in tab_of_interest}
    
    return results

if __name__ == "__main__":

    # Define tabs to parse
    tabs_to_parse = ["General information", "Inputs", "Outputs"]
    
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

    # Export unique flows data to CSV
    unique_flows = parsed_data['unique_flows']
    unique_flows_df = pd.DataFrame()
    for tab_name, flows in unique_flows.items():
        unique_flows_df[tab_name] = pd.Series(flows)
    
    unique_flows_path = os.path.join(OUTPUT_DIR, "unique_flows.csv")
    unique_flows_df.to_csv(unique_flows_path, index=False)
    print(f"\nExported unique flows data to: {unique_flows_path}")
    
    # Print results summary
    print("\nParsed Data Summary:")
    for file_name, file_data in parsed_data.items():
        print(f"\nFile: {file_name}")
        for tab_name, tab_data in file_data.items():
            print(f"  Tab: {tab_name}")
            print(f"  Number of records: {len(tab_data)}")
