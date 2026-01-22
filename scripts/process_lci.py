#!/usr/bin/env python3
"""
LCI Data Processing Script for Hydrogen Production Pathways

This script processes exported LCI models from openLCA and calculates
total Global Warming Potential (GWP) for each hydrogen production pathway.

Author: Generated for H2-LCA-app
Date: 2025-10-22
"""

import pandas as pd
import pathlib
import zipfile
import os
import logging
from typing import List, Dict, Optional, Tuple
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LCIProcessor:
    """
    Main class for processing LCI data from exported openLCA models.
    """
    
    def __init__(self, input_dir: str = "input"):
        """
        Initialize the LCI processor.
        
        Args:
            input_dir: Directory containing LCI Excel files
        """
        self.input_dir = pathlib.Path(input_dir)
        self.lci_data = []
        self.recipe_data = None
        self.results = None
        
    def extract_from_zip(self, zip_path: str) -> None:
        """
        Extract LCI files from a zip archive.
        
        Args:
            zip_path: Path to the zip file containing LCI models
        """
        logger.info(f"Extracting LCI files from {zip_path}")
        
        extract_dir = self.input_dir / "exported LCI models"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
            
        logger.info(f"Files extracted to {extract_dir}")
    
    def detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect column names in Excel files and map to standard names.
        
        Args:
            df: DataFrame from Excel file
            
        Returns:
            Dictionary mapping standard names to actual column names
        """
        columns_lower = [col.lower().strip() for col in df.columns]
        
        mapping = {}
        
        # Common variations for each standard column
        flow_variations = ['flow', 'flow name', 'flowname', 'name']
        amount_variations = ['amount', 'quantity', 'value', 'magnitude']
        unit_variations = ['unit', 'units', 'unit name']
        direction_variations = ['direction', 'type', 'input/output', 'io']
        
        # Find best match for each standard column
        for std_name, variations in [
            ('flow_name', flow_variations),
            ('amount', amount_variations),
            ('unit', unit_variations),
            ('direction', direction_variations)
        ]:
            for col in columns_lower:
                if any(var in col for var in variations):
                    mapping[std_name] = df.columns[columns_lower.index(col)]
                    break
        
        return mapping
    
    def process_excel_file(self, file_path: pathlib.Path) -> Optional[pd.DataFrame]:
        """
        Process a single Excel file and extract LCI data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with standardized LCI data or None if processing fails
        """
        try:
            logger.info(f"Processing {file_path.name}")
            
            # Extract pathway name from filename
            pathway_name = file_path.stem.replace('_', ' ').title()
            
            # Read Inputs and Outputs sheets
            inputs_df = None
            outputs_df = None
            
            try:
                inputs_df = pd.read_excel(file_path, sheet_name='Inputs')
                logger.info(f"  Found Inputs sheet with {len(inputs_df)} rows")
            except Exception as e:
                logger.warning(f"  Could not read Inputs sheet: {e}")
            
            try:
                outputs_df = pd.read_excel(file_path, sheet_name='Outputs')
                logger.info(f"  Found Outputs sheet with {len(outputs_df)} rows")
            except Exception as e:
                logger.warning(f"  Could not read Outputs sheet: {e}")
            
            # Process inputs
            processed_data = []
            
            if inputs_df is not None and not inputs_df.empty:
                mapping = self.detect_column_mapping(inputs_df)
                
                for _, row in inputs_df.iterrows():
                    if pd.isna(row.get('Flow', '')) or row.get('Flow', '') == '':
                        continue
                        
                    processed_data.append({
                        'pathway': pathway_name,
                        'flow_name': str(row.get('Flow', '')).strip(),
                        'amount': float(row.get('Amount', 0)) if pd.notna(row.get('Amount', 0)) else 0,
                        'unit': str(row.get('Unit', '')).strip(),
                        'direction': 'input',
                        'category': str(row.get('Category', '')).strip(),
                        'is_reference': row.get('Is reference?', False),
                        'provider': str(row.get('Provider', '')).strip(),
                        'location': str(row.get('Location', '')).strip()
                    })
            
            # Process outputs
            if outputs_df is not None and not outputs_df.empty:
                mapping = self.detect_column_mapping(outputs_df)
                
                for _, row in outputs_df.iterrows():
                    if pd.isna(row.get('Flow', '')) or row.get('Flow', '') == '':
                        continue
                        
                    processed_data.append({
                        'pathway': pathway_name,
                        'flow_name': str(row.get('Flow', '')).strip(),
                        'amount': float(row.get('Amount', 0)) if pd.notna(row.get('Amount', 0)) else 0,
                        'unit': str(row.get('Unit', '')).strip(),
                        'direction': 'output',
                        'category': str(row.get('Category', '')).strip(),
                        'is_reference': row.get('Is reference?', False),
                        'provider': str(row.get('Provider', '')).strip(),
                        'location': str(row.get('Location', '')).strip()
                    })
            
            if processed_data:
                df = pd.DataFrame(processed_data)
                logger.info(f"  Processed {len(df)} flows")
                return df
            else:
                logger.warning(f"  No valid data found in {file_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def load_all_lci_data(self) -> pd.DataFrame:
        """
        Load and process all LCI Excel files.
        
        Returns:
            Combined DataFrame with all LCI data
        """
        logger.info("Loading all LCI data...")
        
        lci_dir = self.input_dir / "exported LCI models"
        if not lci_dir.exists():
            logger.error(f"LCI directory not found: {lci_dir}")
            return pd.DataFrame()
        
        excel_files = list(lci_dir.glob("*.xlsx"))
        # Filter out temporary files
        excel_files = [f for f in excel_files if not f.name.startswith('~')]
        
        logger.info(f"Found {len(excel_files)} Excel files to process")
        
        all_dataframes = []
        
        for file_path in excel_files:
            df = self.process_excel_file(file_path)
            if df is not None:
                all_dataframes.append(df)
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined LCI data: {len(combined_df)} total flows")
            self.lci_data = combined_df
            return combined_df
        else:
            logger.error("No LCI data could be processed")
            return pd.DataFrame()
    
    def load_recipe_data(self, recipe_file: str = "ReCiPe2016_CFs_v1.1_20180117_GWP.xlsx") -> pd.DataFrame:
        """
        Load ReCiPe characterization factors.
        
        Args:
            recipe_file: Path to ReCiPe Excel file
            
        Returns:
            DataFrame with characterization factors
        """
        recipe_path = self.input_dir / recipe_file
        
        if not recipe_path.exists():
            logger.warning(f"ReCiPe file not found: {recipe_path}")
            logger.info("Creating sample characterization factors...")
            return self._create_sample_recipe_data()
        
        try:
            logger.info(f"Loading ReCiPe data from {recipe_path}")
            
            # Try different sheet names
            sheet_names = ['Characterisation factors', 'CFs', 'Sheet1', 0]
            recipe_df = None
            
            for sheet_name in sheet_names:
                try:
                    recipe_df = pd.read_excel(recipe_path, sheet_name=sheet_name)
                    logger.info(f"Successfully loaded sheet: {sheet_name}")
                    break
                except:
                    continue
            
            if recipe_df is None:
                logger.error("Could not read ReCiPe file")
                return self._create_sample_recipe_data()
            
            # Clean column names
            recipe_df.columns = recipe_df.columns.str.strip().str.lower()
            
            # Try to identify the correct columns
            flow_col = None
            unit_col = None
            gwp_col = None
            
            for col in recipe_df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['flow', 'name', 'substance']):
                    flow_col = col
                elif any(x in col_lower for x in ['unit', 'units']):
                    unit_col = col
                elif any(x in col_lower for x in ['gwp', 'global warming', 'co2', 'carbon']):
                    gwp_col = col
            
            if not all([flow_col, unit_col, gwp_col]):
                logger.warning("Could not identify all required columns in ReCiPe file")
                logger.info(f"Available columns: {list(recipe_df.columns)}")
                return self._create_sample_recipe_data()
            
            # Rename columns to standard names
            recipe_df = recipe_df.rename(columns={
                flow_col: 'flow_name',
                unit_col: 'unit',
                gwp_col: 'cf_gwp'
            })
            
            # Clean data
            recipe_df = recipe_df[['flow_name', 'unit', 'cf_gwp']].dropna()
            recipe_df['flow_name'] = recipe_df['flow_name'].astype(str).str.strip()
            recipe_df['unit'] = recipe_df['unit'].astype(str).str.strip()
            recipe_df['cf_gwp'] = pd.to_numeric(recipe_df['cf_gwp'], errors='coerce').fillna(0)
            
            logger.info(f"Loaded {len(recipe_df)} characterization factors")
            self.recipe_data = recipe_df
            return recipe_df
            
        except Exception as e:
            logger.error(f"Error loading ReCiPe data: {e}")
            return self._create_sample_recipe_data()
    
    def _create_sample_recipe_data(self) -> pd.DataFrame:
        """
        Create sample characterization factors for common flows.
        
        Returns:
            DataFrame with sample characterization factors
        """
        logger.info("Creating sample characterization factors...")
        
        sample_data = [
            # Common emissions
            ('Carbon dioxide, fossil', 'kg', 1.0),
            ('Carbon dioxide', 'kg', 1.0),
            ('CO2', 'kg', 1.0),
            ('Methane, fossil', 'kg', 25.0),
            ('Nitrous oxide', 'kg', 298.0),
            ('Sulfur dioxide', 'kg', 0.0),  # Not a GHG
            
            # Energy flows (approximate values)
            ('Electricity', 'kWh', 0.5),
            ('Natural gas', 'kg', 2.5),
            ('Natural gas', 'm3', 2.0),
            
            # Materials (approximate values)
            ('Steel', 'kg', 2.5),
            ('Water', 'kg', 0.001),
            ('Water', 'm3', 1.0),
            
            # Hydrogen production specific
            ('Hydrogen', 'kg', 0.0),  # Reference product
            ('Oxygen', 'kg', 0.0),   # Co-product
        ]
        
        recipe_df = pd.DataFrame(sample_data, columns=['flow_name', 'unit', 'cf_gwp'])
        self.recipe_data = recipe_df
        logger.info(f"Created {len(recipe_df)} sample characterization factors")
        return recipe_df
    
    def calculate_gwp(self) -> pd.DataFrame:
        """
        Calculate total GWP for each pathway.
        
        Returns:
            DataFrame with pathway GWP results
        """
        if self.lci_data.empty:
            logger.error("No LCI data available for GWP calculation")
            return pd.DataFrame()
        
        if self.recipe_data is None:
            logger.error("No characterization factors available")
            return pd.DataFrame()
        
        logger.info("Calculating GWP...")
        
        # Merge LCI data with characterization factors
        merged = self.lci_data.merge(
            self.recipe_data, 
            on=['flow_name', 'unit'], 
            how='left'
        )
        
        # Calculate GWP contribution for each flow
        merged['gwp_kgco2e'] = merged['amount'] * merged['cf_gwp'].fillna(0)
        
        # Filter out reference products (they don't contribute to GWP)
        merged = merged[merged['is_reference'] != True]
        
        # Aggregate by pathway
        results = merged.groupby('pathway').agg({
            'gwp_kgco2e': 'sum',
            'flow_name': 'count'  # Count of flows
        }).reset_index()
        
        results.columns = ['pathway', 'total_gwp_kgco2e', 'num_flows']
        
        # Sort by GWP (highest first)
        results = results.sort_values('total_gwp_kgco2e', ascending=False)
        
        logger.info(f"Calculated GWP for {len(results)} pathways")
        self.results = results
        return results
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the analysis.
        
        Returns:
            String containing the summary report
        """
        if self.results is None or self.results.empty:
            return "No results available for summary report."
        
        report = []
        report.append("=" * 60)
        report.append("HYDROGEN PRODUCTION PATHWAYS - GWP ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Total pathways analyzed: {len(self.results)}")
        report.append(f"Total LCI flows processed: {len(self.lci_data)}")
        report.append(f"Characterization factors used: {len(self.recipe_data)}")
        report.append("")
        
        report.append("PATHWAY RANKING (by GWP):")
        report.append("-" * 40)
        for i, (_, row) in enumerate(self.results.iterrows(), 1):
            report.append(f"{i:2d}. {row['pathway']:<35} {row['total_gwp_kgco2e']:>8.2f} kg CO₂ eq")
        
        report.append("")
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)
        
        if len(self.results) > 0:
            best_pathway = self.results.iloc[-1]  # Lowest GWP
            worst_pathway = self.results.iloc[0]  # Highest GWP
            
            report.append(f"• Best pathway: {best_pathway['pathway']} ({best_pathway['total_gwp_kgco2e']:.2f} kg CO₂ eq)")
            report.append(f"• Worst pathway: {worst_pathway['pathway']} ({worst_pathway['total_gwp_kgco2e']:.2f} kg CO₂ eq)")
            
            if len(self.results) > 1:
                improvement = ((worst_pathway['total_gwp_kgco2e'] - best_pathway['total_gwp_kgco2e']) / 
                             worst_pathway['total_gwp_kgco2e'] * 100)
                report.append(f"• Improvement potential: {improvement:.1f}%")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Main function to run the LCI processing pipeline.
    """
    logger.info("Starting LCI processing pipeline...")
    
    # Initialize processor
    processor = LCIProcessor()
    
    # Load LCI data
    lci_data = processor.load_all_lci_data()
    if lci_data.empty:
        logger.error("Failed to load LCI data. Exiting.")
        return
    
    # Load characterization factors
    recipe_data = processor.load_recipe_data()
    if recipe_data.empty:
        logger.error("Failed to load characterization factors. Exiting.")
        return
    
    # Calculate GWP
    results = processor.calculate_gwp()
    if results.empty:
        logger.error("Failed to calculate GWP. Exiting.")
        return
    
    # Generate and print summary report
    report = processor.generate_summary_report()
    print(report)
    
    # Save results
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)
    
    results.to_csv(output_dir / "pathway_gwp_results.csv", index=False)
    lci_data.to_csv(output_dir / "all_lci_data.csv", index=False)
    recipe_data.to_csv(output_dir / "characterization_factors.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("LCI processing pipeline completed successfully!")


if __name__ == "__main__":
    main()

