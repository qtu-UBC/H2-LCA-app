import matplotlib.pyplot as plt
import pandas as pd

def generate_impact_piechart(results_df: pd.DataFrame) -> plt:
    """
    Generates a pie chart visualization of impact results by category.
    
    Args:
        results_df: DataFrame containing calculated impact results with 'Category' and 'Calculated Result' columns
        
    Returns:
        matplotlib.pyplot figure object containing the pie chart
    """
    # Create a copy of the DataFrame and extract first category level
    df_copy = results_df.copy()
    df_copy['Category'] = df_copy['Category'].str.split('/').str[0]
    
    # Group results by category and sum impacts
    category_impacts = df_copy.groupby('Category')['Calculated Result'].sum()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate pie chart
    wedges, texts, autotexts = ax.pie(
        category_impacts.values,
        labels=category_impacts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title
    plt.title('Impact Distribution by Category')
    
    # Add legend
    plt.legend(
        wedges, 
        category_impacts.index,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    return plt
