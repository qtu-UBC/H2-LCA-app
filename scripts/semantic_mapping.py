"""
Semantic Table Mapping Script using Ollama

This script automates the semantic similarity mapping between unique flow locations
and processes from the IDEMAT datasheet using Ollama for semantic similarity matching.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama server: ollama serve
    3. Pull an embedding model: ollama pull nomic-embed-text
    
    Alternative embedding models:
    - ollama pull all-minilm
    - ollama pull mxbai-embed-large

Usage:
    python scripts/semantic_mapping.py
    
    Or from project root:
    python -m scripts.semantic_mapping

Output:
    - output/semantic_similarity_table.csv: Mapping table (same format as mockup)
    - output/semantic_similarity_table_with_scores.csv: Includes similarity scores
"""

import pandas as pd
import requests
import json
import logging
import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import time
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    UNIQUE_FLOWS_PROVIDERS_FILE,
    IDEMAT_SHEET,
    INPUT_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    OLLAMA_API_URL,
    OLLAMA_MODEL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ollama API configuration
# Note: Not all Ollama models support embeddings. 
# Models that typically support embeddings: nomic-embed-text, all-minilm, etc.
# For models without embeddings, you may need to use a different approach.
# Configuration is loaded from config.config (can be overridden with environment variables)


class SemanticMapper:
    """Class to handle semantic similarity mapping using Ollama."""
    
    def __init__(self, ollama_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL):
        """
        Initialize the semantic mapper.
        
        Args:
            ollama_url: URL for Ollama API
            model: Ollama model to use for embeddings
        """
        self.ollama_url = ollama_url
        self.model = model
        self.processes = []
        self.process_embeddings = {}
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            # Use the base URL from config (without /api/embeddings)
            base_url = self.ollama_url.replace("/api/embeddings", "")
            tags_url = f"{base_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            if response.status_code == 200:
                logger.info("✓ Ollama is running and accessible")
                return True
            else:
                logger.error(f"✗ Ollama returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("✗ Cannot connect to Ollama. Please make sure Ollama is running.")
            logger.info("  Start Ollama with: ollama serve")
            return False
        except Exception as e:
            logger.error(f"✗ Error checking Ollama connection: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a text using Ollama.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None if error
        """
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")
                if embedding and isinstance(embedding, list):
                    return embedding
                else:
                    logger.warning(f"No valid embedding in response for '{text}'")
                    return None
            else:
                logger.warning(f"Failed to get embedding for '{text}': {response.status_code}")
                logger.warning(f"Response: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout getting embedding for '{text}'")
            return None
        except Exception as e:
            logger.warning(f"Error getting embedding for '{text}': {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def load_processes(self, idemat_file: str) -> List[str]:
        """
        Load all processes from IDEMAT datasheet.
        
        Args:
            idemat_file: Path to IDEMAT Excel file
            
        Returns:
            List of process names
        """
        try:
            logger.info(f"Loading processes from {idemat_file}")
            df = pd.read_excel(idemat_file)
            
            if 'Process' not in df.columns:
                logger.error("'Process' column not found in IDEMAT datasheet")
                return []
            
            # Get unique, non-null processes
            processes = df['Process'].dropna().unique().tolist()
            processes = [str(p).strip() for p in processes if str(p).strip()]
            
            logger.info(f"Loaded {len(processes)} unique processes")
            return processes
            
        except Exception as e:
            logger.error(f"Error loading processes: {e}")
            return []
    
    def get_idemat_file_hash(self, idemat_file: str) -> str:
        """
        Generate a hash of the IDEMAT file to detect changes.
        
        Args:
            idemat_file: Path to IDEMAT Excel file
            
        Returns:
            MD5 hash string of the file
        """
        try:
            with open(idemat_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.warning(f"Could not hash IDEMAT file: {e}")
            return ""
    
    def get_cache_path(self, idemat_file: str) -> Path:
        """
        Get the cache file path for process embeddings.
        
        Args:
            idemat_file: Path to IDEMAT Excel file
            
        Returns:
            Path to cache file
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        file_hash = self.get_idemat_file_hash(idemat_file)
        cache_filename = f"process_embeddings_{file_hash[:8]}_{self.model.replace('/', '_')}.pkl"
        return Path(CACHE_DIR) / cache_filename
    
    def load_cached_embeddings(self, idemat_file: str) -> Optional[dict]:
        """
        Load cached process embeddings if they exist and are valid.
        
        Args:
            idemat_file: Path to IDEMAT Excel file
            
        Returns:
            Dictionary of process embeddings or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(idemat_file)
        
        if not cache_path.exists():
            logger.info("No cached embeddings found. Will compute new embeddings.")
            return None
        
        try:
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache is for current model
            if cached_data.get('model') != self.model:
                logger.info(f"Cache is for different model ({cached_data.get('model')} vs {self.model}). Will recompute.")
                return None
            
            embeddings = cached_data.get('embeddings', {})
            logger.info(f"✓ Loaded {len(embeddings)} cached process embeddings")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {e}. Will recompute.")
            return None
    
    def save_embeddings_cache(self, embeddings: dict, idemat_file: str) -> None:
        """
        Save process embeddings to cache file.
        
        Args:
            embeddings: Dictionary of process embeddings
            idemat_file: Path to IDEMAT Excel file
        """
        cache_path = self.get_cache_path(idemat_file)
        
        try:
            cache_data = {
                'model': self.model,
                'embeddings': embeddings,
                'timestamp': time.time(),
                'idemat_file': idemat_file
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"✓ Saved {len(embeddings)} process embeddings to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Error saving embeddings cache: {e}")
    
    def precompute_process_embeddings(self, processes: List[str], idemat_file: str = None) -> dict:
        """
        Precompute embeddings for all processes to speed up matching.
        Uses cache if available, otherwise computes and saves new embeddings.
        
        Args:
            processes: List of process names
            idemat_file: Path to IDEMAT Excel file (for cache management)
            
        Returns:
            Dictionary mapping process names to embeddings
        """
        # Try to load from cache first
        if idemat_file:
            cached_embeddings = self.load_cached_embeddings(idemat_file)
            if cached_embeddings is not None:
                # Verify all processes are in cache
                missing_processes = [p for p in processes if p not in cached_embeddings]
                if not missing_processes:
                    logger.info("✓ All process embeddings found in cache!")
                    return cached_embeddings
                else:
                    logger.info(f"Cache missing {len(missing_processes)} processes. Computing missing embeddings...")
                    # Compute missing embeddings
                    for i, process in enumerate(missing_processes):
                        embedding = self.get_embedding(process)
                        if embedding:
                            cached_embeddings[process] = embedding
                        time.sleep(0.1)
                    # Save updated cache
                    self.save_embeddings_cache(cached_embeddings, idemat_file)
                    return cached_embeddings
        
        # No cache available, compute all embeddings
        logger.info(f"Computing embeddings for {len(processes)} processes (this may take several minutes)...")
        logger.info("💡 Tip: Embeddings will be cached for future use!")
        embeddings = {}
        
        for i, process in enumerate(processes):
            if i % 50 == 0:
                logger.info(f"  Processing {i+1}/{len(processes)}... ({i+1}/{len(processes)} = {100*(i+1)/len(processes):.1f}%)")
            
            embedding = self.get_embedding(process)
            if embedding:
                embeddings[process] = embedding
            else:
                logger.warning(f"  Failed to get embedding for: {process}")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        logger.info(f"✓ Computed embeddings for {len(embeddings)} processes")
        
        # Save to cache for future use
        if idemat_file:
            self.save_embeddings_cache(embeddings, idemat_file)
        
        return embeddings
    
    def find_most_similar_process(
        self, 
        flow_name: str, 
        process_embeddings: dict
    ) -> Tuple[str, float]:
        """
        Find the most similar process for a given flow name.
        
        Args:
            flow_name: Name of the flow to match
            process_embeddings: Dictionary of process embeddings
            
        Returns:
            Tuple of (most_similar_process, similarity_score)
        """
        if not process_embeddings:
            return ("", 0.0)
        
        # Get embedding for flow name
        flow_embedding = self.get_embedding(flow_name)
        if not flow_embedding:
            logger.warning(f"Could not get embedding for flow: {flow_name}")
            return ("", 0.0)
        
        # Calculate similarity with all processes
        best_match = ""
        best_score = -1.0
        
        for process, process_embedding in process_embeddings.items():
            if not process_embedding:
                continue
            try:
                similarity = self.cosine_similarity(flow_embedding, process_embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_match = process
            except Exception as e:
                logger.warning(f"Error calculating similarity for {process}: {e}")
                continue
        
        return (best_match, best_score)
    
    def load_unique_flows_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load unique flows and providers from a DataFrame.
        
        Args:
            df: DataFrame containing flows to extract
            
        Returns:
            DataFrame with unique flows and locations
        """
        try:
            # Extract all unique flow names from different columns
            unique_flows = set()
            flow_location_pairs = []
            
            # Check for different column name patterns
            flow_columns = [col for col in df.columns if 'flow' in col.lower() or 'Flow' in col]
            provider_columns = [col for col in df.columns if 'provider' in col.lower() or 'Provider' in col]
            location_columns = [col for col in df.columns if 'location' in col.lower() or 'Location' in col]
            
            logger.info(f"Found columns: flows={flow_columns}, providers={provider_columns}, locations={location_columns}")
            
            # Collect flows from input flows column
            if flow_columns:
                for col in flow_columns:
                    for idx, flow in df[col].items():
                        if pd.notna(flow) and str(flow).strip():
                            flow_name = str(flow).strip()
                            location = ""
                            if location_columns:
                                location = df[location_columns[0]].iloc[idx] if pd.notna(df[location_columns[0]].iloc[idx]) else ""
                            
                            unique_flows.add(flow_name)
                            flow_location_pairs.append({
                                'Unique Flow Name': flow_name,
                                'Location': location if location else ""
                            })
            
            # Collect flows from provider columns
            if provider_columns:
                for col in provider_columns:
                    for idx, provider in df[col].items():
                        if pd.notna(provider) and str(provider).strip():
                            provider_name = str(provider).strip()
                            location = ""
                            if location_columns:
                                location = df[location_columns[0]].iloc[idx] if pd.notna(df[location_columns[0]].iloc[idx]) else ""
                            
                            unique_flows.add(provider_name)
                            flow_location_pairs.append({
                                'Unique Flow Name': provider_name,
                                'Location': location if location else ""
                            })
            
            # Remove duplicates while preserving location information
            seen = set()
            unique_pairs = []
            for pair in flow_location_pairs:
                key = (pair['Unique Flow Name'], pair['Location'])
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append(pair)
            
            result_df = pd.DataFrame(unique_pairs)
            logger.info(f"Extracted {len(result_df)} unique flow-location pairs")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error loading unique flows from DataFrame: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def load_unique_flows(self, flows_file: str) -> pd.DataFrame:
        """
        Load unique flows and providers from CSV file.
        
        Args:
            flows_file: Path to unique flows CSV file
            
        Returns:
            DataFrame with unique flows
        """
        try:
            logger.info(f"Loading unique flows from {flows_file}")
            df = pd.read_csv(flows_file)
            
            # Extract all unique flow names from different columns
            unique_flows = set()
            flow_location_pairs = []
            
            # Check for different column name patterns
            flow_columns = [col for col in df.columns if 'flow' in col.lower() or 'Flow' in col]
            provider_columns = [col for col in df.columns if 'provider' in col.lower() or 'Provider' in col]
            location_columns = [col for col in df.columns if 'location' in col.lower() or 'Location' in col]
            
            logger.info(f"Found columns: flows={flow_columns}, providers={provider_columns}, locations={location_columns}")
            
            # Collect flows from input flows column
            if flow_columns:
                for col in flow_columns:
                    for idx, flow in df[col].items():
                        if pd.notna(flow) and str(flow).strip():
                            flow_name = str(flow).strip()
                            location = ""
                            if location_columns:
                                location = df[location_columns[0]].iloc[idx] if pd.notna(df[location_columns[0]].iloc[idx]) else ""
                            
                            unique_flows.add(flow_name)
                            flow_location_pairs.append({
                                'Unique Flow Name': flow_name,
                                'Location': location if location else ""
                            })
            
            # Collect flows from provider columns
            if provider_columns:
                for col in provider_columns:
                    for idx, provider in df[col].items():
                        if pd.notna(provider) and str(provider).strip():
                            provider_name = str(provider).strip()
                            location = ""
                            if location_columns:
                                location = df[location_columns[0]].iloc[idx] if pd.notna(df[location_columns[0]].iloc[idx]) else ""
                            
                            unique_flows.add(provider_name)
                            flow_location_pairs.append({
                                'Unique Flow Name': provider_name,
                                'Location': location if location else ""
                            })
            
            # Remove duplicates while preserving location information
            seen = set()
            unique_pairs = []
            for pair in flow_location_pairs:
                key = (pair['Unique Flow Name'], pair['Location'])
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append(pair)
            
            result_df = pd.DataFrame(unique_pairs)
            logger.info(f"Extracted {len(result_df)} unique flow-location pairs")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error loading unique flows: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_mapping_table(
        self,
        flows_df: pd.DataFrame,
        processes: List[str],
        output_file: str,
        idemat_file: str = None
    ) -> pd.DataFrame:
        """
        Create semantic similarity mapping table.
        
        Args:
            flows_df: DataFrame with unique flows and locations
            processes: List of available processes
            output_file: Path to output CSV file
            idemat_file: Path to IDEMAT file (for cache management)
            
        Returns:
            DataFrame with mapping results
        """
        logger.info("Starting semantic similarity mapping...")
        
        # Precompute process embeddings (uses cache if available)
        process_embeddings = self.precompute_process_embeddings(processes, idemat_file)
        
        if not process_embeddings:
            logger.error("No process embeddings computed. Cannot create mapping.")
            return pd.DataFrame()
        
        # Create mapping results
        mapping_results = []
        
        for idx, row in flows_df.iterrows():
            flow_name = row['Unique Flow Name']
            location = row.get('Location', '')
            
            logger.info(f"Mapping: {flow_name} (Location: {location})")
            
            # Find most similar process
            similar_process, similarity_score = self.find_most_similar_process(
                flow_name,
                process_embeddings
            )
            
            mapping_results.append({
                'Unique Flow Name': flow_name,
                'Location': location if location else "",
                'Most Similar Process': similar_process,
                'Similarity Score': similarity_score
            })
            
            logger.info(f"  → {similar_process} (similarity: {similarity_score:.4f})")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Create DataFrame
        mapping_df = pd.DataFrame(mapping_results)
        
        # Save to CSV (without similarity score column for final output)
        output_df = mapping_df[['Unique Flow Name', 'Location', 'Most Similar Process']].copy()
        output_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved mapping table to {output_file}")
        
        # Also save with similarity scores for reference
        scores_file = output_file.replace('.csv', '_with_scores.csv')
        mapping_df.to_csv(scores_file, index=False)
        logger.info(f"✓ Saved mapping table with scores to {scores_file}")
        
        return output_df


def main():
    """Main function to run the semantic mapping."""
    logger.info("=" * 60)
    logger.info("Semantic Table Mapping using Ollama")
    logger.info("=" * 60)
    
    # Initialize mapper
    mapper = SemanticMapper()
    
    # Check Ollama connection
    if not mapper.check_ollama_connection():
        logger.error("Please start Ollama before running this script.")
        logger.info("Install: https://ollama.ai")
        logger.info("Start: ollama serve")
        logger.info("Pull embedding model: ollama pull nomic-embed-text")
        return
    
    # Check if model is available
    try:
        base_url = mapper.ollama_url.replace("/api/embeddings", "")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if mapper.model not in model_names:
                logger.warning(f"Model '{mapper.model}' not found. Available models: {model_names}")
                logger.info(f"To install: ollama pull {mapper.model}")
                logger.info("Continuing anyway - script will attempt to use the model...")
    except Exception as e:
        logger.warning(f"Could not check available models: {e}")
    
    # Load processes from IDEMAT datasheet
    processes = mapper.load_processes(IDEMAT_SHEET)
    if not processes:
        logger.error("No processes loaded. Cannot continue.")
        return
    
    # Load unique flows
    flows_df = mapper.load_unique_flows(UNIQUE_FLOWS_PROVIDERS_FILE)
    if flows_df.empty:
        logger.error("No unique flows loaded. Cannot continue.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create mapping table
    output_file = os.path.join(OUTPUT_DIR, "semantic_similarity_table.csv")
    mapping_df = mapper.create_mapping_table(flows_df, processes, output_file, IDEMAT_SHEET)
    
    if not mapping_df.empty:
        logger.info("=" * 60)
        logger.info("Mapping completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total mappings: {len(mapping_df)}")
        logger.info("=" * 60)
    else:
        logger.error("Mapping failed. No results generated.")


if __name__ == "__main__":
    main()

