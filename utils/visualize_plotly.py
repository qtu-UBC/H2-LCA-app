"""
Professional Plotly-based visualizations for impact results.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


def prepare_category_data(results_df: pd.DataFrame) -> pd.Series:
    """
    Prepare category data from results DataFrame.
    
    Args:
        results_df: DataFrame with impact results
        
    Returns:
        Series with category impacts
    """
    df_copy = results_df.copy()
    
    # Always use Contribution Category if it exists in the DataFrame
    if 'Contribution Category' in df_copy.columns:
        category_col = 'Contribution Category'
        
        # Keep original values, only process NaN/empty values
        df_copy[category_col] = df_copy[category_col].fillna('').astype(str).str.strip()
        
        # Handle only truly empty values - replace with Category as fallback
        if 'Category' in df_copy.columns:
            mask_to_replace = (df_copy[category_col] == '') | (df_copy[category_col].str.lower() == 'nan')
            if mask_to_replace.any():
                df_copy.loc[mask_to_replace, category_col] = df_copy.loc[mask_to_replace, 'Category'].fillna('').astype(str).str.strip()
        else:
            df_copy.loc[df_copy[category_col] == '', category_col] = 'Uncategorized'
            df_copy.loc[df_copy[category_col].str.lower() == 'nan', category_col] = 'Uncategorized'
    else:
        category_col = 'Category'
        if category_col in df_copy.columns and df_copy[category_col].dtype == 'object':
            df_copy[category_col] = df_copy[category_col].astype(str).str.split('/').str[0]
    
    if category_col not in df_copy.columns:
        category_col = 'Category' if 'Category' in df_copy.columns else df_copy.columns[0]
    
    if 'Calculated Result' not in df_copy.columns:
        raise ValueError("Missing 'Calculated Result' column in results DataFrame")
    
    # Group and sum
    category_impacts = df_copy.groupby(category_col, dropna=False)['Calculated Result'].sum().sort_values(ascending=False)
    
    if len(category_impacts) == 0:
        raise ValueError("No impact values found to visualize")
    
    # Replace any negative values with 0
    category_impacts = category_impacts.clip(lower=0)
    
    return category_impacts


def generate_impact_barchart_plotly(results_df: pd.DataFrame):
    """
    Generates a professional interactive bar chart using Plotly.
    
    Args:
        results_df: DataFrame containing calculated impact results
        
    Returns:
        Tuple of (Plotly figure object, color mapping dict)
    """
    category_impacts = prepare_category_data(results_df)
    
    # Create color scale for professional look
    colors_list = px.colors.qualitative.Set3[:len(category_impacts)]
    
    # Create color mapping for legend
    color_mapping = {label: color for label, color in zip(category_impacts.index, colors_list)}
    
    fig = go.Figure(data=[
        go.Bar(
            x=category_impacts.index,
            y=category_impacts.values,
            marker=dict(
                color=colors_list,
                line=dict(color='#1f1f1f', width=2),
                opacity=0.9
            ),
            # Remove text from bars - use hover only
            hovertemplate='<b>%{x}</b><br>Impact: %{y:.3f} kg CO₂ eq<br>Percentage: %{customdata:.2f}%<extra></extra>',
            customdata=[(val / category_impacts.sum() * 100) if category_impacts.sum() > 0 else 0 for val in category_impacts.values],
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Impact Distribution by Contribution Category',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title='',
            titlefont=dict(size=16, color='#2c3e50'),
            showgrid=False,
            showticklabels=False  # Hide category labels on x-axis
        ),
        yaxis=dict(
            title='CO₂ eq Impact (kg CO₂ eq)',
            titlefont=dict(size=18, color='#2c3e50'),
            tickfont=dict(size=14, color='#34495e'),
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,  # Slightly smaller chart
        margin=dict(l=80, r=50, t=100, b=50),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig, color_mapping


def generate_impact_piechart_plotly(results_df: pd.DataFrame):
    """
    Generates a professional interactive pie chart using Plotly.
    
    Args:
        results_df: DataFrame containing calculated impact results
        
    Returns:
        Tuple of (Plotly figure object, color mapping dict)
    """
    category_impacts = prepare_category_data(results_df)
    
    values = category_impacts.values
    labels = category_impacts.index.tolist()
    
    # Calculate percentages
    total = values.sum()
    percentages = (values / total * 100) if total > 0 else [0] * len(values)
    
    # Create professional color palette
    colors_list = px.colors.qualitative.Set3[:len(labels)]
    
    # Create color mapping for legend
    color_mapping = {label: color for label, color in zip(labels, colors_list)}
    
    # Create hover text with detailed information
    hover_text = [
        f'<b>{label}</b><br>'
        f'Value: {val:.3f} kg CO₂ eq<br>'
        f'Percentage: {pct:.2f}%'
        for label, val, pct in zip(labels, values, percentages)
    ]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=None,  # Remove labels from pie slices
            values=values,
            hole=0.4,  # Donut chart for modern look
            marker=dict(
                colors=colors_list,
                line=dict(color='#ffffff', width=3)
            ),
            textinfo='none',  # No text on slices
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            pull=[0.05 if pct < 1 else 0 for pct in percentages]  # Slight pull for small slices
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Impact Distribution by Contribution Category',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,  # Much larger chart
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=False,  # Legend will be shown below chart
        annotations=[
            dict(
                text=f'<b>Total</b><br>{total:.2f}<br>kg CO₂ eq',
                x=0.5,
                y=0.5,
                font_size=20,
                font_color='#2c3e50',
                showarrow=False
            )
        ]
    )
    
    return fig, color_mapping


def generate_impact_linechart_plotly(results_df: pd.DataFrame):
    """
    Generates a professional interactive line chart using Plotly.
    
    Args:
        results_df: DataFrame containing calculated impact results
        
    Returns:
        Tuple of (Plotly figure object, color mapping dict)
    """
    category_impacts = prepare_category_data(results_df)
    
    # Sort for better visualization
    category_impacts = category_impacts.sort_values(ascending=True)
    
    # Create color scale for markers
    colors_list = px.colors.qualitative.Set3[:len(category_impacts)]
    color_mapping = {label: color for label, color in zip(category_impacts.index, colors_list)}
    
    fig = go.Figure()
    
    # Create individual traces for each category with different colors
    for i, (label, value) in enumerate(category_impacts.items()):
        fig.add_trace(go.Scatter(
            x=[label],
            y=[value],
            mode='markers+lines',
            name=label,
            line=dict(color=colors_list[i], width=3),
            marker=dict(
                size=12,
                color=colors_list[i],
                line=dict(color='#ffffff', width=2)
            ),
            hovertemplate=f'<b>{label}</b><br>Impact: {value:.3f} kg CO₂ eq<extra></extra>',
        ))
    
    # Connect all points with a line
    fig.add_trace(go.Scatter(
        x=category_impacts.index,
        y=category_impacts.values,
        mode='lines',
        name='Trend',
        line=dict(color='#3498db', width=2, dash='dash'),
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>Impact: %{y:.3f} kg CO₂ eq<extra></extra>',
    ))
    
    fig.update_layout(
        title={
            'text': 'Impact Distribution by Contribution Category',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title='',
            titlefont=dict(size=16, color='#2c3e50'),
            showgrid=False,
            showticklabels=False  # Hide category labels on x-axis
        ),
        yaxis=dict(
            title='CO₂ eq Impact (kg CO₂ eq)',
            titlefont=dict(size=18, color='#2c3e50'),
            tickfont=dict(size=14, color='#34495e'),
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,  # Slightly smaller chart
        margin=dict(l=80, r=50, t=100, b=50),
        showlegend=False,  # Legend will be shown below chart
        hovermode='closest'
    )
    
    return fig, color_mapping

