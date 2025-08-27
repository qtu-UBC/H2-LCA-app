"""
This config file contains the following information:
	- path to the H2 manufacturing LCI exported from OpenLCA
	- path to the idemat datasheet
	- path to the mapping file (between naming conventions of ecoinvent and idemat)
"""
import os

# Get the absolute path to the project root (parent of config directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define base input directory relative to project root
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")

# Define paths to input files and folders
IDEMAT_SHEET = os.path.join(INPUT_DIR, "idemat_datasheet_speacial_made.xlsx")
H2_LCI_FOLDER = os.path.join(INPUT_DIR, "exported LCI models")
MAPPING_FILE = os.path.join(INPUT_DIR, "mockup_semantic_similarity_table.csv")
INPUTS_FILE = os.path.join(INPUT_DIR, "inputs.csv") # need to be updated every time parser is run
OUTPUTS_FILE = os.path.join(INPUT_DIR, "outputs.csv") # need to be updated every time parser is run
UNIQUE_FLOWS_PROVIDERS_FILE = os.path.join(INPUT_DIR, "unique_flows_and_providers.csv") # need to be updated every time parser is run
GENERAL_INFO_FILE = os.path.join(INPUT_DIR, "general_information.csv") # need to be updated every time parser is run
RECIPE_CF_FILE = os.path.join(INPUT_DIR, "ReCiPe2016_CFs_v1.1_20180117_GWP.xlsx")


# Define output directory relative to project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Define the path to the log folder
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
