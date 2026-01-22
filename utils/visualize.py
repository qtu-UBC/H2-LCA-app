import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def generate_impact_barchart(results_df: pd.DataFrame) -> plt:
    """
    Generates a bar chart visualization of impact results by contribution category.
    
    Args:
        results_df: DataFrame containing calculated impact results with 'Contribution Category' (or 'Category' as fallback) and 'Calculated Result' columns
        
    Returns:
        matplotlib.pyplot figure object containing the bar chart
    """
    # Create a copy of the DataFrame
    df_copy = results_df.copy()
    
    # Always use Contribution Category if it exists in the DataFrame
    if 'Contribution Category' in df_copy.columns:
        category_col = 'Contribution Category'
        
        # Keep original values, only process NaN/empty values
        # Convert NaN to string 'nan' for easier checking
        df_copy[category_col] = df_copy[category_col].fillna('').astype(str).str.strip()
        
        # Handle only truly empty values - replace with Category as fallback
        if 'Category' in df_copy.columns:
            # Only replace rows where Contribution Category is empty or 'nan' string
            mask_to_replace = (df_copy[category_col] == '') | (df_copy[category_col].str.lower() == 'nan')
            if mask_to_replace.any():
                df_copy.loc[mask_to_replace, category_col] = df_copy.loc[mask_to_replace, 'Category'].fillna('').astype(str).str.strip()
        else:
            df_copy.loc[df_copy[category_col] == '', category_col] = 'Uncategorized'
            df_copy.loc[df_copy[category_col].str.lower() == 'nan', category_col] = 'Uncategorized'
        
    else:
        category_col = 'Category'
        # Extract first category level if using Category column
        if category_col in df_copy.columns and df_copy[category_col].dtype == 'object':
            df_copy[category_col] = df_copy[category_col].astype(str).str.split('/').str[0]
    
    # Group results by contribution category and sum impacts
    # Make sure we're grouping by the correct column
    if category_col not in df_copy.columns:
        # Fallback to Category if Contribution Category was processed incorrectly
        category_col = 'Category' if 'Category' in df_copy.columns else df_copy.columns[0]
    
    # Ensure we have Calculated Result column
    if 'Calculated Result' not in df_copy.columns:
        raise ValueError("Missing 'Calculated Result' column in results DataFrame")
    
    # Group and sum - this is where the actual grouping happens
    category_impacts = df_copy.groupby(category_col, dropna=False)['Calculated Result'].sum().sort_values(ascending=False)
    
    # Don't filter out zero values - keep all categories to show the full picture
    # Only filter if we have no data at all
    if len(category_impacts) == 0:
        raise ValueError("No impact values found to visualize")
    
    # Replace any negative values with 0 (shouldn't happen, but just in case)
    category_impacts = category_impacts.clip(lower=0)
    
    # Show all categories even if some have zero values - user should see the full breakdown
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate bar chart
    bars = ax.bar(
        category_impacts.index,
        category_impacts.values,
        color='steelblue',
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Contribution Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('CO₂ eq Impact (kg CO₂ eq)', fontsize=12, fontweight='bold')
    ax.set_title('Impact Distribution by Contribution Category', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt

def generate_impact_piechart(results_df: pd.DataFrame) -> plt:
    """
    Generates a pie chart visualization of impact results by contribution category.
    
    Args:
        results_df: DataFrame containing calculated impact results with 'Contribution Category' (or 'Category' as fallback) and 'Calculated Result' columns
        
    Returns:
        matplotlib.pyplot figure object containing the pie chart
    """
    # Create a copy of the DataFrame
    df_copy = results_df.copy()
    
    # Always use Contribution Category if it exists in the DataFrame
    if 'Contribution Category' in df_copy.columns:
        category_col = 'Contribution Category'
        
        # Keep original values, only process NaN/empty values
        # Convert NaN to string 'nan' for easier checking
        df_copy[category_col] = df_copy[category_col].fillna('').astype(str).str.strip()
        
        # Handle only truly empty values - replace with Category as fallback
        if 'Category' in df_copy.columns:
            # Only replace rows where Contribution Category is empty or 'nan' string
            mask_to_replace = (df_copy[category_col] == '') | (df_copy[category_col].str.lower() == 'nan')
            if mask_to_replace.any():
                df_copy.loc[mask_to_replace, category_col] = df_copy.loc[mask_to_replace, 'Category'].fillna('').astype(str).str.strip()
        else:
            df_copy.loc[df_copy[category_col] == '', category_col] = 'Uncategorized'
            df_copy.loc[df_copy[category_col].str.lower() == 'nan', category_col] = 'Uncategorized'
        
    else:
        category_col = 'Category'
        # Extract first category level if using Category column
        if category_col in df_copy.columns and df_copy[category_col].dtype == 'object':
            df_copy[category_col] = df_copy[category_col].astype(str).str.split('/').str[0]
    
    # Group results by contribution category and sum impacts
    # Make sure we're grouping by the correct column
    if category_col not in df_copy.columns:
        # Fallback to Category if Contribution Category was processed incorrectly
        category_col = 'Category' if 'Category' in df_copy.columns else df_copy.columns[0]
    
    # Ensure we have Calculated Result column
    if 'Calculated Result' not in df_copy.columns:
        raise ValueError("Missing 'Calculated Result' column in results DataFrame")
    
    # Group and sum - this is where the actual grouping happens
    category_impacts = df_copy.groupby(category_col, dropna=False)['Calculated Result'].sum().sort_values(ascending=False)
    
    # Don't filter out zero values - keep all categories to show the full picture
    # Only filter if we have no data at all
    if len(category_impacts) == 0:
        raise ValueError("No impact values found to visualize")
    
    # Replace any negative values with 0 (shouldn't happen, but just in case)
    category_impacts = category_impacts.clip(lower=0)
    
    # Show all categories even if some have zero values - user should see the full breakdown
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate pie chart
    # Handle very small values - ensure they're displayed
    values = category_impacts.values
    labels = category_impacts.index.tolist()
    
    # Calculate percentages for display
    total = values.sum()
    percentages = (values / total * 100) if total > 0 else [0] * len(values)
    
    # Generate pie chart with all categories - matplotlib will handle zero values
    # Use explode to separate slices for better visibility
    # Fix divide-by-zero: check total > 0 before dividing
    if total > 0:
        explode = [0.05 if val == 0 or val/total < 0.01 else 0 for val in values]
    else:
        explode = [0.05] * len(values)  # All slices exploded if total is 0
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'{pct:.2f}%' if pct > 0.01 else '<0.01%',
        startangle=90,
        pctdistance=0.85,
        explode=explode if any(explode) else None,
        textprops={'fontsize': 10}
    )
    
    # Make sure labels are visible even for very small slices
    for i, (autotext, pct) in enumerate(zip(autotexts, percentages)):
        if pct < 0.1:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title
    plt.title('Impact Distribution by Contribution Category', fontsize=14, fontweight='bold')
    
    # Add legend with values
    legend_labels = [f"{label}: {val:.3f} kg CO₂ eq ({pct:.2f}%)"
                    for label, val, pct in zip(category_impacts.index, values, percentages)]
    plt.legend(
        wedges, 
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9
    )
    
    return plt
