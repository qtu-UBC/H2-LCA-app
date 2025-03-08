


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
    
    # Set up logging
    from config.config import LOG_DIR
    log_file = os.path.join(LOG_DIR, 'parser_errors.log')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Reset log file by opening in write mode
    open(log_file, 'w').close()
    
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
                    try:
                        df = pd.read_excel(file_path, sheet_name=tab)
                    except ValueError as e:
                        error_msg = f"Tab {tab} not found in file {file}"
                        with open(log_file, 'a') as f:
                            f.write(f"{error_msg}\n")
                        continue
                    except pd.errors.EmptyDataError as e:
                        error_msg = f"Empty data in tab {tab} in file {file}"
                        with open(log_file, 'a') as f:
                            f.write(f"{error_msg}\n")
                        continue
                    except PermissionError as e:
                        error_msg = f"Permission denied accessing file {file}"
                        with open(log_file, 'a') as f:
                            f.write(f"{error_msg}\n")
                        continue
                    except Exception as e:
                        if "style" not in str(e).lower():  # Ignore style-related warnings
                            error_msg = f"Error reading tab {tab} in file {file}: {str(e)}"
                            with open(log_file, 'a') as f:
                                f.write(f"{error_msg}\n")
                            continue
                    
                    # Remove empty rows
                    df = df.dropna(how='all')
                    
                    # If tab is Inputs or Outputs, collect unique flows
                    if tab in ['Inputs', 'Outputs']:
                        if 'Flow' in df.columns:
                            unique_flows[tab].update(df['Flow'].unique())
                            
                    # If tab is Providers, collect unique providers and locations
                    if tab == 'Providers':
                        providers_log_file = os.path.join(LOG_DIR, 'providers_data.log')
                        
                        # Create/reset the providers log file
                        open(providers_log_file, 'w').close()
                        
                        if 'Name' in df.columns:
                            providers_list = df['Name'].dropna().unique()
                            unique_providers[tab].update(providers_list)
                            with open(providers_log_file, 'a') as f:
                                f.write(f"\nUnique providers found in {file}:\n")
                                for provider in providers_list:
                                    f.write(f"- {provider}\n")
                        else:
                            with open(providers_log_file, 'a') as f:
                                f.write(f"\nWarning: 'Name' column not found in {file}\n")
                            
                        if 'Location' in df.columns:
                            locations_list = df['Location'].dropna().unique()
                            unique_locations[tab].update(locations_list)
                            with open(providers_log_file, 'a') as f:
                                f.write(f"\nUnique locations found in {file}:\n")
                                for location in locations_list:
                                    f.write(f"- {location}\n")
                        else:
                            with open(providers_log_file, 'a') as f:
                                f.write(f"\nWarning: 'Location' column not found in {file}\n")
                            
                    # Convert DataFrame to dictionary
                    tab_dict = df.to_dict('records')
                    
                    # Store in results dictionary using file name and tab as keys
                    file_key = os.path.splitext(file)[0]
                    if file_key not in results:
                        results[file_key] = {}
                    results[file_key][tab] = tab_dict
                    
                except Exception as e:
                    error_msg = f"Error processing tab {tab} in file {file}: {str(e)}"
                    with open(log_file, 'a') as f:
                        f.write(f"{error_msg}\n")
         
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

    # Export merged data for each tab to separate CSV files and one Excel file
    merged_data = parsed_data['merged_data']
    
    # Export to Excel with each tab
    excel_path = os.path.join(OUTPUT_DIR, "merged_data.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for tab_name, tab_df in merged_data.items():
            tab_df.to_excel(writer, sheet_name=tab_name, index=False)
    print(f"\nExported all merged data to Excel: {excel_path}")
    
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
    
    print("\nUnique Providers:")
    for tab, providers in unique_providers.items():
        print(f"\n{tab}:")
        for provider in providers:
            print(f"  {provider}")
    
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
