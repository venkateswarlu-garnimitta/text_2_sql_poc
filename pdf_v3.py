import os
import time
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from io import BytesIO
import pandas as pd
import numpy as np
import sqlparse
import re
from PIL import Image as PILImage
import requests
from urllib.request import urlopen
from reportlab.platypus import FrameBreak, Frame, PageTemplate
import sys # Added for traceback in error handling

class PDFReportGenerator:
    """
    Class responsible for generating PDF reports containing query information,
    SQL query, data results, visualizations, and a summary.
    """
    
    @staticmethod
    def _format_currency(value, for_table_display=False, for_chart_axis_label=False, for_chart_hover=False) -> str:
        """
        Formats a numerical value into international standard (K, M, B) with optional 'AED' suffix.
        Handles non-numeric values gracefully.
        """
        if pd.isna(value) or not pd.api.types.is_numeric_dtype(type(value)):
            return str(value) # Return as string if not a number or NaN

        # Convert to float to handle potential string numbers that were not caught earlier
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return str(value)

        abs_value = abs(numeric_value)
        suffix = ""
        divisor = 1

        if abs_value >= 1_000_000_000:
            suffix = "B"
            divisor = 1_000_000_000
        elif abs_value >= 1_000_000:
            suffix = "M"
            divisor = 1_000_000
        elif abs_value >= 1_000:
            suffix = "K"
            divisor = 1_000

        formatted_num = numeric_value / divisor

        if for_table_display:
            # For table display: 1K, 22M (no AED)
            if formatted_num == int(formatted_num):
                return f"{int(formatted_num):,}{suffix}"
            else:
                return f"{formatted_num:,.1f}{suffix}" # .1f for conciseness in table
        elif for_chart_axis_label:
            # For chart axis labels: 1.5M, 250K (no AED, concise)
            if formatted_num == int(formatted_num):
                return f"{int(formatted_num):,}{suffix}"
            else:
                return f"{formatted_num:,.1f}{suffix}" # .1f for conciseness on axis
        elif for_chart_hover:
            # For chart hover: 1,234,567.89 AED (full precision with AED)
            return f"{numeric_value:,.2f} AED"
        else:
            # Default for general use, similar to table but can be adjusted
            if formatted_num == int(formatted_num):
                return f"{int(formatted_num):,}{suffix} AED"
            else:
                return f"{formatted_num:,.2f}{suffix} AED"

    @staticmethod
    def _first_page_only(canvas, doc):
        """
        Custom header and footer for only the first page of the PDF.
        Includes left and right logos, and a centered title.
        """
        canvas.saveState()
        styles = getSampleStyleSheet()
        
        # Define custom styles if they are not already globally defined or accessible
        title_style = ParagraphStyle(
            'HeaderTitle', 
            parent=styles['h1'],
            fontSize=18,
            alignment=1,  # Center alignment
            textColor=colors.black 
        )

        timestamp_style = ParagraphStyle(
            'HeaderTimestamp', 
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.gray,
            alignment=1  # Center alignment
        )

        # Define file paths for logos - updated to use jpg files from output directory
        left_logo_path = 'output/left_logo.jpg' 
        right_logo_path = 'output/right_logo.jpg'
        
        # Function to load and resize image, with fallback to internet
        def get_image_buffer(image_path, target_height=0.6*inch):  # Modified to take target_height
            target_width = 2.5 * target_height  # Width is 2.5 times height
            
            if not os.path.exists(image_path):
                print(f"Warning: Logo not found at {image_path}. Attempting to use placeholder from internet.")
                # Try to fetch a placeholder image from the internet
                try:
                    if "left" in image_path:
                        placeholder_url = f"https://via.placeholder.com/{int(target_width*96)}x{int(target_height*96)}.jpg?text=LeftLogo"
                    else:
                        placeholder_url = f"https://via.placeholder.com/{int(target_width*96)}x{int(target_height*96)}.jpg?text=RightLogo"
                    
                    response = requests.get(placeholder_url)
                    img_buffer = BytesIO(response.content)
                    img = PILImage.open(img_buffer)
                except Exception as e:
                    print(f"Error fetching internet placeholder: {e}")
                    # Create a simple blank image as a last resort
                    img_buffer = BytesIO()
                    blank_img = PILImage.new('RGB', (int(target_width*96), int(target_height*96)), color='lightgray') 
                    blank_img.save(img_buffer, format='JPEG')
                    img_buffer.seek(0)
                    return img_buffer
            else:
                try:
                    img = PILImage.open(image_path)
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
                    # Create a blank image if can't open
                    img_buffer = BytesIO()
                    blank_img = PILImage.new('RGB', (int(target_width*96), int(target_height*96)), color='lightgray') 
                    blank_img.save(img_buffer, format='JPEG')
                    img_buffer.seek(0)
                    return img_buffer
            
            # Use LANCZOS for high-quality downsampling and maintain aspect ratio implicitly by setting new dimensions
            img = img.resize((int(target_width*96), int(target_height*96)), PILImage.LANCZOS) 
            
            # Save as PNG for potentially better quality and transparency support (if original has it)
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG', dpi=(300, 300)) # Increased DPI for clarity
            img_buffer.seek(0)
            return img_buffer

        # Load logo buffers with new target dimensions
        left_logo_buffer = get_image_buffer(left_logo_path, target_height=0.6*inch) 
        right_logo_buffer = get_image_buffer(right_logo_path, target_height=0.6*inch) 

        # Calculate coordinates relative to the page top
        page_width, page_height = letter

        # Y position for header content - moved up since we've removed margins
        header_content_y_start = page_height - 0.5*inch

        # Add left logo
        if left_logo_buffer:
            img = Image(left_logo_buffer, width=2.5*0.6*inch, height=0.6*inch) # Set width to 2.5*height (2.5 * 0.6 = 1.5)
            img_width, img_height = img.wrapOn(canvas, doc.width, doc.topMargin) 
            img.drawOn(canvas, doc.leftMargin, header_content_y_start - img_height) 

        # Add right logo
        if right_logo_buffer:
            img = Image(right_logo_buffer, width=2.5*0.6*inch, height=0.6*inch) # Set width to 2.5*height (2.5 * 0.6 = 1.5)
            img_width, img_height = img.wrapOn(canvas, doc.width, doc.topMargin) 
            img.drawOn(canvas, doc.width + doc.leftMargin - img_width, header_content_y_start - img_height) 

        # Add title (Nl2SQL Query Report) in the center
        title_text = "NL2SQL Query Report"
        p = Paragraph(title_text, title_style)
        p_width, p_height = p.wrapOn(canvas, doc.width, doc.topMargin)
        p.drawOn(canvas, doc.leftMargin + (doc.width - p_width) / 2.0, header_content_y_start - p_height)

        # Add timestamp in footer
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"Generated on: {timestamp}"
        p_timestamp = Paragraph(footer_text, timestamp_style)
        p_ts_width, p_ts_height = p_timestamp.wrapOn(canvas, doc.width, doc.bottomMargin)
        p_timestamp.drawOn(canvas, doc.leftMargin + (doc.width - p_ts_width) / 2.0, 0.3*inch)
        
        canvas.restoreState()
    
    @staticmethod
    def _later_pages(canvas, doc):
        """
        Empty function for later pages - no header/footer
        """
        pass

    @staticmethod
    def generate_report(
        query: str, 
        sql_query: str, 
        result_df: pd.DataFrame,
        summary: str = None,
        output_path: str = None
    ) -> str:
        
        """
        Generates a comprehensive PDF report with query details, summary, and visualizations.
        
        Args:
            query (str): The original natural language query
            sql_query (str): The generated SQL query
            result_df (pd.DataFrame): The query result data
            summary (str): Markdown-formatted summary of the data analysis
            output_path (str, optional): Path where to save the PDF. If None, generates in memory.
            
        Returns:
            str: Path to the generated PDF file or bytes if output_path is None
        """
        # Create a BytesIO object to store PDF in memory if no output path
        if output_path is None: 
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                    rightMargin=36, leftMargin=36,
                                    topMargin=40, bottomMargin=30) # Reduced margins for more space
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                    rightMargin=36, leftMargin=36,
                                    topMargin=40, bottomMargin=30) # Reduced margins for more space
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Define custom styles and add them to the stylesheet
        normal_style = ParagraphStyle(
            'CustomNormal', 
            parent=styles['Normal'],
            fontSize=10,
            leading=14,  # Increased leading for better spacing
            spaceBefore=3,
            spaceAfter=5
        )
        styles.add(normal_style) 

        title_style = ParagraphStyle(
            'CustomTitle', 
            parent=styles['h1'], 
            fontSize=18,
            alignment=1,  # Center alignment
            spaceAfter=12,
            textColor=colors.black 
        )
        styles.add(title_style) 
        
        heading2_style = ParagraphStyle(
            'CustomHeading2', 
            parent=styles['h2'], 
            fontSize=14,
            textColor=colors.black, 
            spaceBefore=12,
            spaceAfter=8
        )
        styles.add(heading2_style) 
        
        heading3_style = ParagraphStyle(
            'CustomHeading3', 
            parent=styles['h3'], 
            fontSize=12,
            textColor=colors.black, 
            spaceBefore=2,
            spaceAfter=2
        )
        styles.add(heading3_style) 
        
        code_style = ParagraphStyle(
            'CodeStyle', 
            parent=styles['Code'], 
            fontName='Courier',
            fontSize=9,
            leading=12,
            leftIndent=10,
            rightIndent=10,
            backColor=colors.lightgrey,
            borderPadding=5,
            spaceAfter=10
        )
        styles.add(code_style) 
        
        summary_style = ParagraphStyle(
            'SummaryStyle', 
            parent=normal_style,
            fontSize=10,
            textColor=colors.black,
            leading=16,  # Increased leading for better spacing
            spaceBefore=6,
            spaceAfter=8
        )
        styles.add(summary_style) 
        
        note_style = ParagraphStyle(
            'Note', 
            parent=normal_style, 
            textColor=colors.red,
            fontSize=8,
            leading=10
        )
        styles.add(note_style) 

        styles.add(ParagraphStyle(
            name='BulletStyle',
            parent=normal_style, 
            fontSize=10,
            textColor=colors.black,
            leading=16,  # Increased leading for better spacing
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=18,
            firstLineIndent=-18,
            bulletIndent=0,
            bulletFontName='Helvetica-Bold',
            bulletFontSize=10,
            bulletColor=colors.black
        ))

        table_cell_style = ParagraphStyle(
            'TableCellStyle',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=4, # TA_LEFT or TA_CENTER
            wordWrap='LTR' 
        )
        styles.add(table_cell_style)
        
        # Container for PDF elements
        elements = []
        
        elements.append(Spacer(1, 0.8*inch)) # Add more space below header on first page

        # Add Query Section
        elements.append(Paragraph("Natural Language Query", styles['CustomHeading2'])) 
        elements.append(Paragraph(query, styles['CustomNormal'])) 
        
        # Add SQL Query Section with reduced spacing
        elements.append(Paragraph("Generated SQL Query", styles['CustomHeading2'])) 
        formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
        elements.append(Paragraph(formatted_sql, styles['CodeStyle'])) 
        
        # Make a copy of result_df for processing to avoid modifying original
        df_processed = result_df.copy()

        # --- Identify and Convert Potential Numeric/Currency Columns ---
        # Iterate through object/string columns to try and convert them
        for col in df_processed.select_dtypes(include=['object', 'string']).columns:
            # Check if all non-NaN values can be converted to numeric
            temp_numeric_series = pd.to_numeric(df_processed[col], errors='coerce')
            if temp_numeric_series.notna().all(): # If all non-null values are convertible
                df_processed[col] = temp_numeric_series # Perform the conversion
            else:
                # If not fully numeric, check if any non-null string values contain currency symbols
                contains_currency_symbol = False
                for val in df_processed[col].dropna().head(10): # Check first 10 non-null values
                    if isinstance(val, str) and ('AED' in val or '€' in val or '$' in val or '£' in val or '₹' in val):
                        contains_currency_symbol = True
                        break
                if contains_currency_symbol:
                    # Attempt to clean and convert if symbols are found
                    df_processed[col] = df_processed[col].astype(str).str.replace(r'[^\d.-]', '', regex=True) # Remove all non-numeric chars except dot and minus
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Get the list of all numeric columns after initial conversions
        numeric_cols_current = df_processed.select_dtypes(include=['number']).columns

        # --- Robust Currency Column Identification Logic ---
        identified_currency_cols = set()

        # 1. Keywords in column names
        keyword_currency_cols = [
            col for col in numeric_cols_current
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'price', 'total', 'revenue', 'cost', 'spend', 'budget', 'sales', 'profit', 'aed'])
        ]
        identified_currency_cols.update(keyword_currency_cols)

        # 2. Numeric Suffix Check (e.g., _2024, _24234)
        numeric_suffix_pattern = re.compile(r'_\d+$') # Matches _ followed by one or more digits at the end
        suffix_currency_cols = [
            col for col in numeric_cols_current
            if numeric_suffix_pattern.search(col)
        ]
        identified_currency_cols.update(suffix_currency_cols)

        # 3. Large Integer Value Heuristic
        LARGE_VALUE_THRESHOLD = 1000 

        for col in numeric_cols_current:
            if col not in identified_currency_cols: # Only check columns not already identified
                if not df_processed[col].empty and df_processed[col].count() > 0:
                    non_na_values = df_processed[col].dropna()
                    if not non_na_values.empty:
                        is_integer_series = np.isclose(non_na_values, non_na_values.astype(int))
                        
                        if is_integer_series.all(): # If all non-NaN values are integers
                            count_large_integers = (non_na_values >= LARGE_VALUE_THRESHOLD).sum()
                            percentage_large_integers = count_large_integers / len(non_na_values)

                            if percentage_large_integers > 0.5: # More than 50% are large integers
                                identified_currency_cols.add(col)
        
        # Final list of currency columns for formatting
        currency_like_cols = list(identified_currency_cols)
        
        # Add Results Section (if data exists)
        if df_processed is not None and not df_processed.empty:
            elements.append(Paragraph("Query Results", styles['CustomHeading2'])) 
            elements.append(Paragraph(f"Number of rows: {len(df_processed)} | Number of columns: {len(df_processed.columns)}", 
                                     styles['CustomNormal'])) 
            
            preview_df = df_processed.head(10) # Use df_processed here
            
            table_data = [preview_df.columns.tolist()]  # Header row
            for i, row in preview_df.iterrows():
                formatted_row = []
                for col_name, val in row.items():
                    if col_name in currency_like_cols: # Use the identified currency columns
                        formatted_row.append(Paragraph(PDFReportGenerator._format_currency(val, for_table_display=True), styles['TableCellStyle']))
                    else:
                        formatted_row.append(Paragraph(str(val), styles['TableCellStyle']))
                table_data.append(formatted_row)
            
            if table_data:
                available_width = 7.5 * inch 
                col_widths = [max(0.8*inch, available_width / len(table_data[0]))] * len(table_data[0])
                data_table = Table(table_data, repeatRows=1, colWidths=col_widths)
                
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), # Changed to darkblue for better contrast
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'), # Header Center
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'), # Content Left
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('TOPPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8), 
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightsteelblue]), # Better color scheme
                ])
                data_table.setStyle(table_style)
                
                elements.append(data_table)
                
                if len(preview_df.columns) > 5:
                    elements.append(Paragraph("Note: Preview showing first 10 rows. Values may be wrapped within columns.", 
                                             styles['Note'])) 
            
            if summary:
                elements.append(Paragraph("Data Analysis Summary", styles['CustomHeading2'])) 
                
                def parse_markdown_to_flowables(md_text, styles):
                    flowables = []
                    md_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md_text)
                    md_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', md_text)
                    md_text = re.sub(r'_(.*?)_', r'<i>\1</i>', md_text)
                    
                    lines = md_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            flowables.append(Spacer(1, 0.1*inch))  # Add space between paragraphs
                            continue

                        if line.startswith('### '):
                            flowables.append(Paragraph(line[4:].strip(), styles['CustomHeading3']))
                        elif line.startswith('## '):
                            flowables.append(Paragraph(line[3:].strip(), styles['CustomHeading2']))
                        elif line.startswith('# '):
                            flowables.append(Paragraph(line[2:].strip(), styles['CustomTitle']))
                        elif line.startswith('* ') or line.startswith('- '):
                            flowables.append(Paragraph(f'<bullet>&bull;</bullet>{line[2:].strip()}', styles['BulletStyle']))
                        else:
                            flowables.append(Paragraph(line, styles['SummaryStyle']))
                    return flowables
                
                summary_flowables = parse_markdown_to_flowables(summary, styles)
                elements.extend(summary_flowables)
            
            if not df_processed.empty: # Use df_processed here
                elements.append(PageBreak())
                elements.append(Paragraph("Visualizations", styles['CustomHeading2'])) 
                elements.append(Spacer(1, 0.1*inch))
                
                vis_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap"]
                
                # Improved color maps for better visualizations
                color_maps = {
                    "Bar Chart": "viridis",
                    "Line Chart": "tab10",
                    "Scatter Plot": "plasma",
                    "Pie Chart": "Set3",
                    "Heatmap": "coolwarm"
                }
                
                charts_to_add = [] 
                for vis_type in vis_types:
                    try:
                        img_data = PDFReportGenerator._create_visualization_in_memory(
                            df_processed, # Pass df_processed
                            vis_type, 
                            currency_like_cols, # Pass currency_like_cols
                            color_map=color_maps.get(vis_type)
                        )
                        
                        if img_data:
                            title_para = Paragraph(f"{vis_type}", styles['CustomHeading3'])
                            # Adjusted image width and height for vertical stacking
                            # Using the same size as before for other charts, but may be adjusted for pie chart in _create_visualization_in_memory
                            img = Image(img_data, width=5.5*inch, height=3.8*inch)  
                            charts_to_add.append([title_para, img]) # Store title and image as a unit
                            
                    except Exception as e:
                        print(f"Error adding {vis_type} visualization: {str(e)}")
                        continue
                
                # Place charts vertically, two per page
                chart_elements = []
                
                if len(charts_to_add) > 0:
                    for i in range(0, len(charts_to_add), 2):
                        # Create a list for the elements of the current page (up to two charts)
                        page_charts = []
                        
                        # Add the first chart (title + image)
                        page_charts.extend(charts_to_add[i])
                        
                        # Add a spacer between the two charts if a second chart exists
                        if i + 1 < len(charts_to_add):
                            page_charts.append(Spacer(1, 0.3*inch)) # Space between charts
                            page_charts.extend(charts_to_add[i+1])
                        
                        # Create a Table to hold the charts on this page, ensuring they stay together
                        # A single-column table to stack elements vertically, centered
                        chart_table = Table([[elem] for elem in page_charts], colWidths=[doc.width])
                        chart_table.setStyle(TableStyle([
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ]))
                        chart_elements.append(chart_table)
                        
                        # Add page break if there are more charts to follow on the next page
                        if (i + 2) < len(charts_to_add): 
                            chart_elements.append(PageBreak())
                
                elements.extend(chart_elements)

        # Set up the custom page templates for first page and subsequent pages
        first_page_template = PageTemplate(id='FirstPage', frames=[Frame(
            doc.leftMargin, doc.bottomMargin, doc.width, doc.height)], 
            onPage=PDFReportGenerator._first_page_only)
        
        later_pages_template = PageTemplate(id='LaterPages', frames=[Frame(
            doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
            onPage=PDFReportGenerator._later_pages)
        
        doc.addPageTemplates([first_page_template, later_pages_template])
        
        # Set the template to use for each page
        elements.insert(0, Paragraph("", ParagraphStyle("dummy")))  # First page uses FirstPage template
        
        # Build the PDF
        doc.build(elements)
        
        if output_path is None: 
            pdf_bytes = buffer.getvalue()
            buffer.close()
            return pdf_bytes
        
        return output_path
    
    @staticmethod
    def _create_visualization_in_memory(df: pd.DataFrame, vis_type: str, currency_like_cols: list, color_map: str = None) -> BytesIO:
        """
        Creates a visualization and returns it as an in-memory BytesIO object.
        Enhanced with better color maps and styling.
        """
        try:
            img_buffer = BytesIO()
            plt.style.use('seaborn-v0_8-darkgrid')  # Modern, professional style
            
            # Use high-DPI figure for sharper plots, standard size for most charts
            figsize = (6, 4) 
            # Increased figure size for pie chart to accommodate legend better
            if vis_type == "Pie Chart":
                figsize = (7, 5) 
            
            fig, ax = plt.subplots(figsize=figsize, dpi=120) # Create figure and axes here

            if vis_type == "Bar Chart":
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    
                    plot_df = df.groupby(x_col)[y_col].sum().reset_index()

                    if len(plot_df) > 10:
                        plot_df = plot_df.sort_values(by=y_col, ascending=False)
                        top_10 = plot_df.head(10)
                        others_sum = plot_df.iloc[10:][y_col].sum()
                        others_row = pd.DataFrame({x_col: ['Others'], y_col: [others_sum]})
                        plot_df = pd.concat([top_10, others_row], ignore_index=True)

                    if color_map:
                        colors_list = plt.cm.get_cmap(color_map, len(plot_df[x_col]))
                    else:
                        colors_list = plt.cm.get_cmap('viridis', len(plot_df[x_col]))
                        
                    bars = ax.bar(plot_df[x_col], plot_df[y_col], 
                                  color=[colors_list(i) for i in range(len(plot_df[x_col]))], 
                                  alpha=0.85,
                                  edgecolor='white',
                                  linewidth=0.7)
                    
                    for bar in bars:
                        height = bar.get_height()
                        # Use _format_currency for bar labels, with for_chart_axis_label=True
                        ax.annotate(PDFReportGenerator._format_currency(height, for_chart_axis_label=True),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), 
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
                    
                    ax.set_title(f"{y_col} Contribution by {x_col}", fontsize=12) 
                    ax.set_xlabel(x_col, fontsize=10) 
                    
                    # Y-axis label: Add (AED) if it's a currency column
                    y_label_text = y_col
                    if y_col in currency_like_cols:
                        y_label_text = f"{y_col} (AED)"
                    ax.set_ylabel(y_label_text, fontsize=10) 
                    
                    # Y-axis formatter: Use _format_currency with for_chart_axis_label=True if currency
                    if y_col in currency_like_cols:
                        formatter = plt.FuncFormatter(lambda x, pos: PDFReportGenerator._format_currency(x, for_chart_axis_label=True))
                        ax.yaxis.set_major_formatter(formatter)

                    plt.xticks(rotation=45, ha='right', fontsize=8) 
                    plt.yticks(fontsize=8) 
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    plt.close(fig) # Close the specific figure
                    
                    img_buffer.seek(0)
                    return img_buffer
            
            elif vis_type == "Line Chart":
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) >= 1:
                    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    x_col = None
                    if date_cols:
                        for d_col in date_cols: 
                            try:
                                df[d_col] = pd.to_datetime(df[d_col])
                                df = df.sort_values(d_col)
                                x_col = d_col
                                break 
                            except Exception:
                                continue
                    if x_col is None: 
                        x_col = df.columns[0]

                    y_cols_to_plot = numeric_cols[:min(len(numeric_cols), 3)] 
                    
                    if color_map:
                        colors_line = plt.cm.get_cmap(color_map, len(y_cols_to_plot)).colors
                    else:
                        colors_line = plt.cm.get_cmap('tab10', len(y_cols_to_plot)).colors
                        
                    markers = ['o', 's', '^', 'd', 'x', '+', '*']
                    
                    for i, y_col in enumerate(y_cols_to_plot):  
                        ax.plot(df[x_col], df[y_col], 
                                 marker=markers[i % len(markers)], 
                                 color=colors_line[i % len(colors_line)],
                                 linewidth=2, markersize=4, 
                                 alpha=0.8, label=y_col)
                    
                    ax.set_title(f"Trend of {', '.join(y_cols_to_plot)} Over {x_col}", fontsize=12) 
                    ax.set_xlabel(x_col, fontsize=10) 
                    
                    # Y-axis label: Add (AED) if any of the y_cols are currency
                    y_label_text = "Values"
                    if any(col in currency_like_cols for col in y_cols_to_plot):
                        y_label_text = "Values (AED)"
                    ax.set_ylabel(y_label_text, fontsize=10) 

                    # Y-axis formatter: Use _format_currency if any of the y_cols are currency
                    if any(col in currency_like_cols for col in y_cols_to_plot):
                        formatter = plt.FuncFormatter(lambda x, pos: PDFReportGenerator._format_currency(x, for_chart_axis_label=True))
                        ax.yaxis.set_major_formatter(formatter)

                    ax.legend(fontsize=7, loc='best') 
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.xticks(rotation=45, ha='right', fontsize=8) 
                    plt.yticks(fontsize=8) 
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    plt.close(fig) # Close the specific figure
                    
                    img_buffer.seek(0)
                    return img_buffer

            elif vis_type == "Scatter Plot":
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]

                    if len(numeric_cols) >= 3:
                        c_col = numeric_cols[2]
                        
                        if color_map:
                            cmap = plt.cm.get_cmap(color_map)
                        else:
                            cmap = plt.cm.get_cmap('plasma')
                            
                        scatter = ax.scatter(df[x_col], df[y_col], 
                                             c=df[c_col], cmap=cmap, 
                                             alpha=0.8, s=50, edgecolors='w', linewidths=0.5) 
                        plt.colorbar(scatter, ax=ax, label=c_col) # Reduced font size
                    else:
                        # If no third numeric column, use a categorical column for coloring if available
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                        if len(categorical_cols) > 0:
                            cat_col = categorical_cols[0]
                            categories = df[cat_col].unique()
                            
                            if color_map:
                                colors_list = plt.cm.get_cmap(color_map, len(categories))
                            else:
                                colors_list = plt.cm.get_cmap('tab10', len(categories))
                                
                            for i, category in enumerate(categories):
                                subset = df[df[cat_col] == category]
                                ax.scatter(subset[x_col], subset[y_col], 
                                           color=colors_list(i), 
                                           alpha=0.7, s=50, 
                                           edgecolors='w', linewidths=0.5,
                                           label=category)
                            ax.legend(fontsize=7, loc='best') 
                        else:
                            # Simple scatter plot with no coloring
                            ax.scatter(df[x_col], df[y_col], 
                                       alpha=0.7, s=50, color='royalblue', 
                                       edgecolors='w', linewidths=0.5)
                    
                    # Add correlation coefficient
                    corr = df[x_col].corr(df[y_col])
                    ax.annotate(f'Correlation: {corr:.2f}', 
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)) 
                    
                    ax.set_title(f"Relationship Between {x_col} and {y_col}", fontsize=12) 
                    
                    # X-axis label: Add (AED) if it's a currency column
                    x_label_text = x_col
                    if x_col in currency_like_cols:
                        x_label_text = f"{x_col} (AED)"
                    ax.set_xlabel(x_label_text, fontsize=10) 

                    # Y-axis label: Add (AED) if it's a currency column
                    y_label_text = y_col
                    if y_col in currency_like_cols:
                        y_label_text = f"{y_col} (AED)"
                    ax.set_ylabel(y_label_text, fontsize=10) 

                    # X-axis formatter: Use _format_currency if currency
                    if x_col in currency_like_cols:
                        formatter_x = plt.FuncFormatter(lambda x, pos: PDFReportGenerator._format_currency(x, for_chart_axis_label=True))
                        ax.xaxis.set_major_formatter(formatter_x)
                    
                    # Y-axis formatter: Use _format_currency if currency
                    if y_col in currency_like_cols:
                        formatter_y = plt.FuncFormatter(lambda y, pos: PDFReportGenerator._format_currency(y, for_chart_axis_label=True))
                        ax.yaxis.set_major_formatter(formatter_y)

                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.xticks(fontsize=8) 
                    plt.yticks(fontsize=8) 
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    plt.close(fig) # Close the specific figure
                    
                    img_buffer.seek(0)
                    return img_buffer
                
            elif vis_type == "Pie Chart":
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    
                    # Group by the categorical column and sum the numeric column
                    plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                    
                    # Limit to top categories for readability
                    if len(plot_df) > 7:
                        plot_df = plot_df.sort_values(by=y_col, ascending=False)
                        top_6 = plot_df.head(6)
                        others_sum = plot_df.iloc[6:][y_col].sum()
                        others_row = pd.DataFrame({x_col: ['Others'], y_col: [others_sum]})
                        plot_df = pd.concat([top_6, others_row], ignore_index=True)
                    
                    if color_map:
                        colors_pie = plt.cm.get_cmap(color_map, len(plot_df))
                    else:
                        colors_pie = plt.cm.get_cmap('Set3', len(plot_df))
                        
                    wedges, texts, autotexts = ax.pie(
                        plot_df[y_col],
                        labels=None, # Remove labels directly on slices
                        autopct='%1.1f%%',
                        startangle=90,
                        shadow=False,
                        colors=[colors_pie(i) for i in range(len(plot_df))],
                        pctdistance=0.85, # Distance of percentage labels from center
                        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
                    )
                    
                    # Improve autotext appearance
                    for autotext in autotexts:
                        autotext.set_fontsize(8) 
                        autotext.set_weight('bold')
                        autotext.set_color('black')
                    
                    # Create legend with category names and values
                    legend_labels = []
                    for cat, val in zip(plot_df[x_col], plot_df[y_col]):
                        if y_col in currency_like_cols:
                            legend_labels.append(f"{cat} ({PDFReportGenerator._format_currency(val, for_chart_hover=True)})")
                        else:
                            legend_labels.append(f"{cat} ({val:,.2f})") # Default numeric format
                    
                    # Adjusted legend placement for better automatic fitting
                    ax.legend(wedges, legend_labels, 
                              title=x_col, 
                              loc='center left', 
                              bbox_to_anchor=(1, 0.5), 
                              fontsize=8) 
                    
                    ax.set_title(f"Distribution of {y_col} by {x_col}", fontsize=12) 
                    plt.subplots_adjust(right=0.7) 
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    plt.close(fig) # Close the specific figure
                    
                    img_buffer.seek(0)
                    return img_buffer
                
            elif vis_type == "Heatmap":
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) >= 4:
                    # Select first 10 numeric columns max for readability
                    selected_cols = numeric_cols[:min(10, len(numeric_cols))]
                    correlation_matrix = df[selected_cols].corr()
                    
                    if color_map:
                        cmap = color_map
                    else:
                        cmap = 'coolwarm'
                        
                    heatmap = ax.imshow(correlation_matrix, cmap=cmap, interpolation='nearest')
                    
                    # Add colorbar
                    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7) 
                    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15, fontsize=9) 
                    
                    # Add correlation values
                    for i in range(len(correlation_matrix)):
                        for j in range(len(correlation_matrix)):
                            corr_value = correlation_matrix.iloc[i, j]
                            text_color = 'white' if abs(corr_value) > 0.7 else 'black'
                            ax.text(j, i, f'{corr_value:.2f}', 
                                    ha='center', va='center', 
                                    color=text_color, fontsize=7) 
                    
                    # Add column names as ticks
                    column_names = [col[:10] + '...' if len(col) > 10 else col for col in correlation_matrix.columns]
                    ax.set_xticks(np.arange(len(column_names)))
                    ax.set_yticks(np.arange(len(column_names)))
                    ax.set_xticklabels(column_names, rotation=45, ha='right', fontsize=7) 
                    ax.set_yticklabels(column_names, fontsize=7) 
                    
                    ax.set_title('Correlation Heatmap of Numeric Variables', fontsize=12) 
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    plt.close(fig) # Close the specific figure
                    
                    img_buffer.seek(0)
                    return img_buffer
                
            return None
                
        except Exception as e:
            print(f"Error creating {vis_type} visualization: {str(e)}")
            traceback = sys.exc_info()[2]
            print(f"Line: {traceback.tb_lineno}")
            # Ensure the figure is closed even if an error occurs
            if 'fig' in locals() and fig is not None:
                plt.close(fig)
            return None

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create sample dataframe
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Department': ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Operations', 'Sales', 'Marketing', 'IT', 'Finance'],
        'Amount': [500000, 350000, 250000, 150000, 600000, 450000, 700000, 200000, 350000, 550000],
        'Quantity': [5, 3, 7, 2, 6, 4, 8, 5, 4, 7],
        'Revenue_2023': [1000000, 700000, 500000, 300000, 1200000, 900000, 1400000, 400000, 700000, 1100000],
        'Cost_2024': [300000, 200000, 150000, 80000, 400000, 250000, 450000, 120000, 200000, 350000],
        'Profit_Q1': [700000, 500000, 350000, 220000, 800000, 650000, 950000, 280000, 500000, 750000],
        'Employee_Count': [50, 30, 25, 15, 60, 40, 70, 35, 28, 55] # Added a non-currency large integer column
    }
    
    df = pd.DataFrame(data)
    
    query = "Show me the revenue by department"
    sql_query = "SELECT Department, SUM(Revenue) AS Total_Revenue FROM sales_data GROUP BY Department ORDER BY Total_Revenue DESC"
    
    summary = """
    # Revenue Analysis by Department
    
    The data shows revenue distribution across various departments:
    
    * **Sales** has the highest revenue contribution at 2.4M AED (33% of total).
    * **Finance** follows with 1.75M AED (24% of total).
    * **Operations** contributes 0.9M AED (12% of total).
    * **Marketing** generates 1.1M AED (15% of total).
    * **IT** has 1.2M AED (16% of total).
    
    The Finance department shows the highest average transaction value, while Sales has the highest number of transactions.
    
    ### Key Insights
    
    - Three departments (Sales, Finance, IT) account for 73% of total revenue.
    - There's potential for growth in the HR department, which currently has the lowest contribution.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    try:
        # Generate the report
        output_path = PDFReportGenerator.generate_report(
            query=query,
            sql_query=sql_query,
            result_df=df,
            summary=summary,
            output_path='output/sample_report.pdf'
        )
        print(f"Report successfully generated at: {output_path}")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        # Adding traceback for better debugging of user's code
        import traceback
        traceback.print_exc()