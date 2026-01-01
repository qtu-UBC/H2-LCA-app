"""
PDF Export Utility for H2-LCA App
Generates comprehensive PDF reports from the application data
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import pandas as pd
import os
from pathlib import Path


def create_pdf_report(output_path, app_data):
    """
    Create a comprehensive PDF report from the LCA app data
    
    Args:
        output_path: Path where PDF will be saved
        app_data: Dictionary containing all app data to include in report
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    # Title
    story.append(Paragraph("H2 Manufacturing LCI Data Explorer", title_style))
    story.append(Paragraph("Life Cycle Assessment Report", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Selected Source File / Pathway
    if 'selected_source' in app_data and app_data['selected_source']:
        story.append(Paragraph(f"<b>Selected Source File:</b> {app_data['selected_source']}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Selected Pathways
    if 'selected_pathways' in app_data and app_data['selected_pathways']:
        pathways_text = ", ".join(app_data['selected_pathways'])
        story.append(Paragraph(f"<b>Selected Pathways:</b> {pathways_text}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Input Data
    if 'inputs_df' in app_data and not app_data['inputs_df'].empty:
        story.append(Paragraph("Input Data", heading_style))
        inputs_df = app_data['inputs_df']
        
        # Limit columns for PDF readability
        display_cols = ['Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']
        if 'Contribution Category' in inputs_df.columns:
            display_cols.append('Contribution Category')
        
        available_cols = [col for col in display_cols if col in inputs_df.columns]
        inputs_display = inputs_df[available_cols].head(50)  # Limit to 50 rows
        
        # Create table
        table_data = [available_cols]  # Header
        for _, row in inputs_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        story.append(table)
        if len(inputs_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(inputs_df)} rows</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Output Data
    if 'outputs_df' in app_data and not app_data['outputs_df'].empty:
        story.append(Paragraph("Output Data", heading_style))
        outputs_df = app_data['outputs_df']
        
        # Limit columns for PDF readability
        display_cols = ['Flow', 'Category', 'Amount', 'Unit', 'Provider', 'Location']
        if 'Contribution Category' in outputs_df.columns:
            display_cols.append('Contribution Category')
        
        available_cols = [col for col in display_cols if col in outputs_df.columns]
        outputs_display = outputs_df[available_cols].head(50)  # Limit to 50 rows
        
        # Create table
        table_data = [available_cols]  # Header
        for _, row in outputs_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        story.append(table)
        if len(outputs_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(outputs_df)} rows</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Semantic Mapping Results
    if 'semantic_mapping_results' in app_data and app_data['semantic_mapping_results'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("Semantic Flow Mapping Results", heading_style))
        mapping_df = app_data['semantic_mapping_results']
        
        display_cols = ['Unique Flow Name', 'Location', 'Most Similar Process']
        if 'Similarity Score' in mapping_df.columns:
            display_cols.append('Similarity Score')
        
        available_cols = [col for col in display_cols if col in mapping_df.columns]
        mapping_display = mapping_df[available_cols].head(50)
        
        table_data = [available_cols]
        for _, row in mapping_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
        ]))
        
        story.append(table)
        if len(mapping_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(mapping_df)} mappings</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Prepopulated Mappings
    if 'prepopulated_mappings' in app_data and app_data['prepopulated_mappings'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("Prepopulated Mappings from Pathways", heading_style))
        prepop_df = app_data['prepopulated_mappings']
        
        display_cols = ['Flow_Name', 'Location', 'Mapped_Process', 'Source_Pathway']
        available_cols = [col for col in display_cols if col in prepop_df.columns]
        prepop_display = prepop_df[available_cols].head(50)
        
        table_data = [available_cols]
        for _, row in prepop_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(table)
        if len(prepop_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(prepop_df)} prepopulated mappings</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Ollama Mappings
    if 'ollama_mappings' in app_data and app_data['ollama_mappings'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("Ollama-Generated Mappings", heading_style))
        ollama_df = app_data['ollama_mappings']
        
        display_cols = ['Flow_Name', 'Location', 'Mapped_Process', 'Similarity_Score']
        available_cols = [col for col in display_cols if col in ollama_df.columns]
        ollama_display = ollama_df[available_cols].head(50)
        
        table_data = [available_cols]
        for _, row in ollama_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcyan),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(table)
        if len(ollama_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(ollama_df)} Ollama mappings</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Pathway Comparison Results
    if 'pathway_comparison_results' in app_data and app_data['pathway_comparison_results'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("Pathway Comparison Results", heading_style))
        comparison_df = app_data['pathway_comparison_results']
        
        display_cols = ['Flow_Name', 'Location', 'Mapped_Process', 'Mapping_Source']
        if 'Similarity_Score' in comparison_df.columns:
            display_cols.append('Similarity_Score')
        
        available_cols = [col for col in display_cols if col in comparison_df.columns]
        comparison_display = comparison_df[available_cols].head(50)
        
        table_data = [available_cols]
        for _, row in comparison_display.iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row[available_cols]])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgreen]),
        ]))
        
        story.append(table)
        if len(comparison_df) > 50:
            story.append(Paragraph(f"<i>Showing first 50 of {len(comparison_df)} comparison results</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # GWP Analysis Results
    if 'gwp_results' in app_data and app_data['gwp_results'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("GWP Analysis Results", heading_style))
        gwp_df = app_data['gwp_results']
        
        table_data = [list(gwp_df.columns)]
        for _, row in gwp_df.head(30).iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
    
    # Impact Results
    if 'impact_results' in app_data and app_data['impact_results'] is not None:
        story.append(PageBreak())
        story.append(Paragraph("Impact Results", heading_style))
        impact_df = app_data['impact_results']
        
        table_data = [list(impact_df.columns)]
        for _, row in impact_df.head(30).iterrows():
            table_data.append([str(val) if pd.notna(val) else '' for val in row])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("--- End of Report ---", styles['Normal']))
    story.append(Paragraph(f"Generated by H2 Manufacturing LCI Data Explorer", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    return output_path

