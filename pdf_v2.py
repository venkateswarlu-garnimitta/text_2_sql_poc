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
import sys
import functools  # Added for caching with lru_cache

class PDFReportGenerator:
    """
    Class responsible for generating PDF reports containing query information,
    SQL query, data results, visualizations, and a summary.
    """
    
    @staticmethod
    @functools.lru_cache(maxsize=32)  # Cache currency formatting results
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
        
        # Keep a class-level cache of loaded images to avoid reloading
        if not hasattr(PDFReportGenerator, '_image_cache'):
            PDFReportGenerator._image_cache = {}
        
        # Function to load and resize image, with fallback to internet
        def get_image_buffer(image_path, target_height=0.6*inch):
            # Check if image is already in cache
            cache_key = f"{image_path}_{target_height}"
            if cache_key in PDFReportGenerator._image_cache:
                return PDFReportGenerator._image_cache[cache_key]
                
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
                    PDFReportGenerator._image_cache[cache_key] = img_buffer
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
                    PDFReportGenerator._image_cache[cache_key] = img_buffer
                    return img_buffer
            
            # Use LANCZOS for high-quality downsampling and maintain aspect ratio implicitly by setting new dimensions
            img = img.resize((int(target_width*96), int(target_height*96)), PILImage.LANCZOS) 
            
            # Save as PNG for potentially better quality and transparency support (if original has it)
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG', dpi=(300, 300)) # Increased DPI for clarity
            img_buffer.seek(0)
            
            # Cache the result
            PDFReportGenerator._image_cache[cache_key] = img_buffer
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
    def _identify_currency_columns(df_processed):
        """
        Helper method to identify currency columns in the dataframe
        """
        # --- Identify and Convert Potential Numeric/Currency Columns ---
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

        # 1. Keywords in column names - use set operations for faster matching
        currency_keywords = {'amount', 'value', 'price', 'total', 'revenue', 'cost', 'spend', 'budget', 'sales', 'profit', 'aed'}
        for col in numeric_cols_current:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in currency_keywords):
                identified_currency_cols.add(col)

        # 2. Numeric Suffix Check (e.g., _2024, _24234)
        numeric_suffix_pattern = re.compile(r'_\d+$') # Matches _ followed by one or more digits at the end
        for col in numeric_cols_current:
            if numeric_suffix_pattern.search(col):
                identified_currency_cols.add(col)

        # 3. Large Integer Value Heuristic
        LARGE_VALUE_THRESHOLD = 1000 

        for col in numeric_cols_current:
            if col not in identified_currency_cols: # Only check columns not already identified
                if not df_processed[col].empty and df_processed[col].count() > 0:
                    non_na_values = df_processed[col].dropna()
                    if not non_na_values.empty:
                        # Faster method to check for integers
                        is_integer_series = (non_na_values % 1 == 0)
                        
                        if is_integer_series.all(): # If all non-NaN values are integers
                            count_large_integers = (non_na_values >= LARGE_VALUE_THRESHOLD).sum()
                            percentage_large_integers = count_large_integers / len(non_na_values)

                            if percentage_large_integers > 0.5: # More than 50% are large integers
                                identified_currency_cols.add(col)
        
        return list(identified_currency_cols)

    @staticmethod
    def _create_styles():
        """
        Create and return the styles dictionary only once to avoid repetitive style creation
        """
        # Create a styles singleton if it doesn't exist
        if not hasattr(PDFReportGenerator, '_styles_cache'):
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
            
            # Store styles in class variable
            PDFReportGenerator._styles_cache = styles
            
        return PDFReportGenerator._styles_cache

    @staticmethod
    def _create_paragraph_from_text(text, style):
        """
        Create and return a Paragraph object with text and style
        """
        return Paragraph(text, style)

    @staticmethod
    def _parse_markdown_to_flowables(md_text, styles):
        """
        Parse markdown text to flowables
        """
        flowables = []
        # Pre-compile regex patterns for better performance
        bold_pattern = re.compile(r'\*\*(.*?)\*\*')
        italic_pattern = re.compile(r'\*(.*?)\*')
        underscore_pattern = re.compile(r'_(.*?)_')
        
        md_text = bold_pattern.sub(r'<b>\1</b>', md_text)
        md_text = italic_pattern.sub(r'<i>\1</i>', md_text)
        md_text = underscore_pattern.sub(r'<i>\1</i>', md_text)
        
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

    @staticmethod
    def _prepare_visualizations(df_processed, currency_like_cols):
        """
        Prepare all visualizations in parallel to avoid repeated similar operations
        """
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
        
        # Process the dataframe once for visualization data
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        date_cols = [col for col in df_processed.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        # Reuse processed data across visualizations
        processed_data = {
            'categorical_cols': categorical_cols,
            'numeric_cols': numeric_cols,
            'date_cols': date_cols
        }
        
        # Lower DPI for faster rendering while maintaining quality
        render_dpi = 120
        
        for vis_type in vis_types:
            try:
                img_data = PDFReportGenerator._create_visualization_optimized(
                    df_processed,
                    vis_type,
                    currency_like_cols,
                    color_map=color_maps.get(vis_type),
                    processed_data=processed_data,
                    dpi=render_dpi
                )
                
                if img_data:
                    styles = PDFReportGenerator._create_styles()
                    title_para = PDFReportGenerator._create_paragraph_from_text(
                        f"{vis_type}", styles['CustomHeading3']
                    )
                    # Adjusted image width and height for vertical stacking
                    img = Image(img_data, width=5.5*inch, height=3.8*inch)
                    charts_to_add.append([title_para, img])
                    
            except Exception as e:
                print(f"Error adding {vis_type} visualization: {str(e)}")
                continue
                
        return charts_to_add

    @staticmethod
    def _create_visualization_optimized(df, vis_type, currency_like_cols, color_map=None, processed_data=None, dpi=120):
        """
        Optimized visualization creation that reuses pre-processed data when available
        """
        try:
            img_buffer = BytesIO()
            
            # Use a more efficient style setting
            plt.style.use('seaborn-v0_8-darkgrid')  
            
            # Use high-DPI figure for sharper plots, standard size for most charts
            figsize = (6, 4) 
            # Increased figure size for pie chart to accommodate legend better
            if vis_type == "Pie Chart":
                figsize = (7, 5) 
            
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Use pre-processed data if available
            if processed_data:
                categorical_cols = processed_data['categorical_cols']
                numeric_cols = processed_data['numeric_cols']
                date_cols = processed_data['date_cols']
            else:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

            # Use color_map caching to avoid recomputing the same colormap
            if not hasattr(PDFReportGenerator, '_color_map_cache'):
                PDFReportGenerator._color_map_cache = {}
                
            # Get or create color_map
            def get_color_map(map_name, item_count):
                cache_key = f"{map_name}_{item_count}"
                if cache_key not in PDFReportGenerator._color_map_cache:
                    PDFReportGenerator._color_map_cache[cache_key] = plt.cm.get_cmap(map_name, item_count)
                return PDFReportGenerator._color_map_cache[cache_key]

            if vis_type == "Bar Chart":
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    
                    # More efficient grouping with categoricals
                    if df[x_col].dtype == 'object':
                        df[x_col] = df[x_col].astype('category')
                    
                    # Use more efficient aggregation
                    plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()

                    if len(plot_df) > 10:
                        plot_df = plot_df.nlargest(10, y_col)
                        # Set up an "Others" category with all remaining rows
                        others_sum = df.loc[~df[x_col].isin(plot_df[x_col]), y_col].sum()
                        others_row = pd.DataFrame({x_col: ['Others'], y_col: [others_sum]})
                        plot_df = pd.concat([plot_df, others_row], ignore_index=True)

                    # Get color map once and reuse
                    colors_list = get_color_map(color_map or 'viridis', len(plot_df))
                        
                    bars = ax.bar(plot_df[x_col], plot_df[y_col], 
                                  color=[colors_list(i) for i in range(len(plot_df))], 
                                  alpha=0.85,
                                  edgecolor='white',
                                  linewidth=0.7)
                    
                    # Create bar annotations in batch for better performance
                    bar_heights = [bar.get_height() for bar in bars]
                    x_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
                    
                    for i, (height, x_pos) in enumerate(zip(bar_heights, x_positions)):
                        formatted_value = PDFReportGenerator._format_currency(height, for_chart_axis_label=True)
                        ax.annotate(formatted_value,
                                    xy=(x_pos, height),
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
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    img_buffer.seek(0)
                    return img_buffer
            
            elif vis_type == "Line Chart":
                if len(numeric_cols) >= 1:
                    x_col = None
                    # Try to find a date column first
                    if date_cols:
                        for d_col in date_cols: 
                            try:
                                # Only convert if not already datetime
                                if not pd.api.types.is_datetime64_any_dtype(df[d_col]):
                                    df[d_col] = pd.to_datetime(df[d_col])
                                df = df.sort_values(d_col)
                                x_col = d_col
                                break 
                            except Exception:
                                continue
                    if x_col is None: 
                        x_col = df.columns[0]

                    # Limit to top 3 numeric columns
                    y_cols_to_plot = numeric_cols[:min(len(numeric_cols), 3)] 
                    
                    # Get colormap once
                    if color_map:
                        colors_line = get_color_map(color_map, len(y_cols_to_plot)).colors
                    else:
                        colors_line = get_color_map('tab10', len(y_cols_to_plot)).colors
                        
                    markers = ['o', 's', '^', 'd', 'x', '+', '*']
                    
                    # Plot each line
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
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    img_buffer.seek(0)
                    return img_buffer
            
            elif vis_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    # If we have a third numeric column, use it for sizing points
                    size_col = None
                    if len(numeric_cols) >= 3:
                        size_col = numeric_cols[2]
                    
                    # If we have a categorical column, use it for coloring points
                    color_col = None
                    if len(categorical_cols) > 0:
                        color_col = categorical_cols[0]
                    
                    if color_col:
                        unique_categories = df[color_col].nunique()
                        # Limit categories to prevent overcrowding
                        if unique_categories > 10:
                            top_categories = df[color_col].value_counts().nlargest(9).index.tolist()
                            plot_df = df[df[color_col].isin(top_categories)].copy()
                            plot_df.loc[~plot_df[color_col].isin(top_categories), color_col] = 'Other'
                        else:
                            plot_df = df
                            
                        # Create color mapping
                        categories = plot_df[color_col].unique()
                        color_map_obj = get_color_map(color_map or 'tab10', len(categories))
                        color_dict = {cat: color_map_obj(i) for i, cat in enumerate(categories)}
                        
                        # Create scatter with appropriate sizing
                        if size_col:
                            # Normalize size for better visualization
                            size_values = plot_df[size_col].values
                            size_normalized = 20 + (size_values - size_values.min()) * 100 / (size_values.max() - size_values.min() + 1e-10)
                            
                            scatter = ax.scatter(plot_df[x_col], plot_df[y_col], 
                                           c=[color_dict[cat] for cat in plot_df[color_col]],
                                           s=size_normalized, alpha=0.7, edgecolors='w', linewidth=0.5)
                            
                            # Create legend with proxy artists
                            from matplotlib.lines import Line2D
                            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                                     markerfacecolor=color_dict[cat], 
                                                     markersize=8, label=cat) 
                                              for cat in categories]
                            ax.legend(handles=legend_elements, fontsize=7, title=color_col, loc='best')
                        else:
                            # When no sizing column is available
                            for category, color in color_dict.items():
                                mask = plot_df[color_col] == category
                                ax.scatter(plot_df.loc[mask, x_col], plot_df.loc[mask, y_col], 
                                           c=color, s=30, alpha=0.7, edgecolors='w', 
                                           linewidth=0.5, label=category)
                            ax.legend(fontsize=7, title=color_col, loc='best')
                    else:
                        # Simple scatter without categories
                        if size_col:
                            # Normalize size for better visualization
                            size_values = df[size_col].values
                            size_normalized = 20 + (size_values - size_values.min()) * 100 / (size_values.max() - size_values.min() + 1e-10)
                            
                            scatter = ax.scatter(df[x_col], df[y_col], 
                                           c=get_color_map(color_map or 'viridis', 1)(0),
                                           s=size_normalized, alpha=0.7, edgecolors='w', linewidth=0.5)
                            
                            # Add size legend
                            handles, labels = scatter.legend_elements(prop="sizes", num=4, alpha=0.7, 
                                                                     func=lambda s: (s - 20) * (size_values.max() - size_values.min()) / 100 + size_values.min())
                            ax.legend(handles, labels, fontsize=7, title=size_col, loc='best')
                        else:
                            ax.scatter(df[x_col], df[y_col], 
                                      c=get_color_map(color_map or 'viridis', 1)(0),
                                      s=30, alpha=0.7, edgecolors='w', linewidth=0.5)
                    
                    # Set chart labels and formatting
                    ax.set_title(f"Relationship between {x_col} and {y_col}", fontsize=12)
                    
                    # X-axis label: Add (AED) if it's a currency column
                    x_label_text = x_col
                    if x_col in currency_like_cols:
                        x_label_text = f"{x_col} (AED)"
                        formatter = plt.FuncFormatter(lambda x, pos: PDFReportGenerator._format_currency(x, for_chart_axis_label=True))
                        ax.xaxis.set_major_formatter(formatter)
                    ax.set_xlabel(x_label_text, fontsize=10)
                    
                    # Y-axis label: Add (AED) if it's a currency column
                    y_label_text = y_col
                    if y_col in currency_like_cols:
                        y_label_text = f"{y_col} (AED)"
                        formatter = plt.FuncFormatter(lambda x, pos: PDFReportGenerator._format_currency(x, for_chart_axis_label=True))
                        ax.yaxis.set_major_formatter(formatter)
                    ax.set_ylabel(y_label_text, fontsize=10)
                    
                    ax.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    img_buffer.seek(0)
                    return img_buffer
            
            elif vis_type == "Pie Chart":
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    val_col = numeric_cols[0]
                    
                    # Group by the categorical column, creating a more efficient groupby
                    if df[cat_col].dtype == 'object':
                        df[cat_col] = df[cat_col].astype('category')
                        
                    plot_df = df.groupby(cat_col, observed=True)[val_col].sum().reset_index()
                    
                    # Limit to top slices for readability
                    if len(plot_df) > 10:
                        top_df = plot_df.nlargest(9, val_col)
                        others_value = plot_df.loc[~plot_df[cat_col].isin(top_df[cat_col]), val_col].sum()
                        others_df = pd.DataFrame({cat_col: ['Others'], val_col: [others_value]})
                        plot_df = pd.concat([top_df, others_df], ignore_index=True)
                    
                    # Create colormap
                    colors = get_color_map(color_map or 'Set3', len(plot_df))
                    
                    # Sort by value for better visualization
                    plot_df = plot_df.sort_values(val_col, ascending=False)
                    
                    # Create pie chart
                    wedges, texts, autotexts = ax.pie(
                        plot_df[val_col], 
                        labels=None,  # No labels on the pie to avoid overcrowding
                        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                        startangle=90, 
                        colors=[colors(i) for i in range(len(plot_df))],
                        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                        textprops={'fontsize': 8}
                    )
                    
                    # Customize autopct text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    # Add legend outside the pie
                    labels = []
                    for i, (cat, val) in enumerate(zip(plot_df[cat_col], plot_df[val_col])):
                        if val_col in currency_like_cols:
                            formatted_val = PDFReportGenerator._format_currency(val, for_table_display=True)
                            labels.append(f"{cat} ({formatted_val})")
                        else:
                            labels.append(f"{cat} ({val:,.0f})")
                    
                    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
                    
                    ax.set_title(f"Distribution of {val_col} by {cat_col}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    img_buffer.seek(0)
                    return img_buffer
            
            elif vis_type == "Heatmap":
                # Heatmap requires at least 2 categorical and 1 numeric column, or 2 numeric columns
                if (len(categorical_cols) >= 2 and len(numeric_cols) >= 1) or len(numeric_cols) >= 2:
                    # Case 1: Two categorical columns and a numeric value
                    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                        x_col = categorical_cols[0]
                        y_col = categorical_cols[1]
                        value_col = numeric_cols[0]
                        
                        # Create efficient pivot table
                        # Convert to categories first to optimize memory
                        if df[x_col].dtype == 'object':
                            df[x_col] = df[x_col].astype('category')
                        if df[y_col].dtype == 'object':
                            df[y_col] = df[y_col].astype('category')
                            
                        heatmap_df = df.pivot_table(index=y_col, columns=x_col, values=value_col, 
                                                   aggfunc='sum', fill_value=0)
                        
                        # Limit size for readability
                        if heatmap_df.shape[0] > 10 or heatmap_df.shape[1] > 10:
                            # Keep top categories by sum
                            row_sums = heatmap_df.sum(axis=1)
                            col_sums = heatmap_df.sum(axis=0)
                            
                            top_rows = row_sums.nlargest(min(10, len(row_sums))).index
                            top_cols = col_sums.nlargest(min(10, len(col_sums))).index
                            
                            heatmap_df = heatmap_df.loc[top_rows, top_cols]
                        
                    # Case 2: Two numeric columns - create correlation heatmap
                    else:
                        # Only use numeric columns for correlation
                        heatmap_df = df[numeric_cols].corr()
                        
                        # Limit to top correlations if needed
                        if len(numeric_cols) > 10:
                            heatmap_df = heatmap_df.iloc[:10, :10]
                    
                    # Create heatmap using chosen colormap
                    im = ax.imshow(heatmap_df, cmap=color_map or 'coolwarm')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    
                    # Configure labels and ticks
                    ax.set_xticks(np.arange(len(heatmap_df.columns)))
                    ax.set_yticks(np.arange(len(heatmap_df.index)))
                    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=7)
                    ax.set_yticklabels(heatmap_df.index, fontsize=7)
                    
                    # Annotate cells with values for better readability
                    for i in range(len(heatmap_df.index)):
                        for j in range(len(heatmap_df.columns)):
                            value = heatmap_df.iloc[i, j]
                            # Format currency values if applicable
                            if value_col in currency_like_cols if 'value_col' in locals() else False:
                                text = PDFReportGenerator._format_currency(value, for_chart_axis_label=True)
                            elif abs(value) < 0.01:  # For correlation values
                                text = f"{value:.2f}"
                            elif abs(value) < 1000:
                                text = f"{value:.1f}"
                            else:
                                text = f"{value:,.0f}"
                                
                            # Choose text color based on background for readability
                            if 'value_col' in locals():  # For case 1 (value heatmap)
                                color_val = im.norm(value)
                                text_color = 'white' if 0.3 < color_val < 0.8 else 'black'
                            else:  # For case 2 (correlation heatmap)
                                text_color = 'white' if abs(value) > 0.6 else 'black'
                                
                            ax.text(j, i, text, ha="center", va="center", 
                                   fontsize=6, color=text_color)
                    
                    # Set title based on type of heatmap
                    if 'value_col' in locals():
                        ax.set_title(f"Heatmap of {value_col} by {x_col} and {y_col}", fontsize=12)
                    else:
                        ax.set_title("Correlation Heatmap of Numeric Variables", fontsize=12)
                        
                    plt.tight_layout()
                    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    img_buffer.seek(0)
                    return img_buffer
                
            # Return None if no visualization could be created
            return None
            
        except Exception as e:
            print(f"Error in visualization for {vis_type}: {str(e)}")
            return None

    @staticmethod
    def generate_pdf_report(output_file_path, natural_language_query, sql_query, data_df, summary_text, page_size=letter, page_margin=0.5):
        """
        Generate a PDF report based on the given data, query and summary.
        """
        try:
            # Initialize document
            doc = SimpleDocTemplate(
                output_file_path,
                pagesize=page_size,
                leftMargin=page_margin*inch,
                rightMargin=page_margin*inch,
                topMargin=page_margin*inch,
                bottomMargin=page_margin*inch
            )
            
            # Create initial empty story for content flowables
            story = []
            
            # Get styles
            styles = PDFReportGenerator._create_styles()
            
            # Process and add the natural language query section
            nl_query_title = PDFReportGenerator._create_paragraph_from_text("Natural Language Query", styles['CustomHeading2'])
            story.append(nl_query_title)
            story.append(PDFReportGenerator._create_paragraph_from_text(natural_language_query, styles['CustomNormal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Format the SQL query for better presentation
            formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            sql_title = PDFReportGenerator._create_paragraph_from_text("Generated SQL Query", styles['CustomHeading2'])
            story.append(sql_title)
            story.append(PDFReportGenerator._create_paragraph_from_text(formatted_sql, styles['CodeStyle']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add data results section
            data_title = PDFReportGenerator._create_paragraph_from_text("Data Results", styles['CustomHeading2'])
            story.append(data_title)
            
            # Process the data for display
            df_processed = data_df.copy()
            
            # Identify currency columns for proper formatting
            currency_like_cols = PDFReportGenerator._identify_currency_columns(df_processed)
            
            # Create Data Table
            if not df_processed.empty:
                # Limit to 50 rows for PDF readability
                display_df = df_processed.head(50) if len(df_processed) > 50 else df_processed
                
                # Process the data for display                
                data = []
                headers = list(display_df.columns)
                data.append(headers)
                
                # Batch process rows for performance
                for _, row in display_df.iterrows():
                    formatted_row = []
                    for col in headers:
                        cell_value = row[col]
                        if col in currency_like_cols and pd.notna(cell_value):
                            formatted_value = PDFReportGenerator._format_currency(cell_value, for_table_display=True)
                            formatted_row.append(formatted_value)
                        elif pd.api.types.is_numeric_dtype(type(cell_value)):
                            # Format other numbers with commas for readability
                            formatted_row.append(f"{cell_value:,}" if pd.notna(cell_value) else "")
                        else:
                            # For non-numeric data, convert to string and limit length
                            if pd.notna(cell_value):
                                str_value = str(cell_value)
                                if len(str_value) > 50:  # Limit cell text length
                                    formatted_row.append(f"{str_value[:47]}...")
                                else:
                                    formatted_row.append(str_value)
                            else:
                                formatted_row.append("")
                    data.append(formatted_row)
                
                # Calculate optimal column widths based on content
                col_widths = []
                for i in range(len(headers)):
                    max_width = max(len(str(row[i])) for row in data) * 0.09 * inch
                    # Set reasonable min/max width constraints
                    col_widths.append(min(max(max_width, 0.5 * inch), 2.0 * inch))
                
                # Set table style with alternating row colors
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ])
                
                # Add alternating row colors for better readability
                for i in range(1, len(data)):
                    if i % 2 == 0:
                        table_style.add('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)
                
                data_table = Table(data, colWidths=col_widths)
                data_table.setStyle(table_style)
                
                story.append(data_table)
                
                # Add note if data was truncated
                if len(df_processed) > 50:
                    truncation_note = PDFReportGenerator._create_paragraph_from_text(
                        f"Note: Showing first 50 of {len(df_processed)} results.", styles['Note']
                    )
                    story.append(truncation_note)
            else:
                no_data_note = PDFReportGenerator._create_paragraph_from_text(
                    "No data returned for this query.", styles['Note']
                )
                story.append(no_data_note)
            
            story.append(Spacer(1, 0.3*inch))
            
            # Add visualizations section
            vis_title = PDFReportGenerator._create_paragraph_from_text("Data Visualizations", styles['CustomHeading2'])
            story.append(vis_title)
            
            if not df_processed.empty and len(df_processed) > 1:
                charts_to_add = PDFReportGenerator._prepare_visualizations(df_processed, currency_like_cols)
                
                if not charts_to_add:
                    story.append(PDFReportGenerator._create_paragraph_from_text(
                        "No suitable visualizations could be created for this data.", styles['Note']
                    ))
                else:
                    # Add visualizations in a 2x3 layout
                    max_charts = min(len(charts_to_add), 6)  # Limit to 6 charts
                    story.append(PDFReportGenerator._create_paragraph_from_text(
                        "The following visualizations provide insights into your data:", 
                        styles['CustomNormal']
                    ))
                    
                    for i in range(0, max_charts, 2):
                        # Make a mini table of two charts side by side if we have enough
                        if i+1 < max_charts:
                            # 2-column layout with charts
                            chart_table_data = [[charts_to_add[i][0], charts_to_add[i+1][0]], 
                                               [charts_to_add[i][1], charts_to_add[i+1][1]]]
                            chart_table = Table(chart_table_data, colWidths=[3*inch, 3*inch])
                            chart_table.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                                ('TOPPADDING', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                            ]))
                            story.append(chart_table)
                        else:
                            # Single chart centered
                            story.append(charts_to_add[i][0])
                            story.append(charts_to_add[i][1])
            else:
                story.append(PDFReportGenerator._create_paragraph_from_text(
                    "No visualizations available due to insufficient data.", styles['Note']
                ))
            
            story.append(Spacer(1, 0.2*inch))
            story.append(PageBreak())
            
            # Add insights and summary section
            summary_title = PDFReportGenerator._create_paragraph_from_text("Summary and Insights", styles['CustomHeading2'])
            story.append(summary_title)
            
            # Process markdown summary if available
            if summary_text:
                summary_flowables = PDFReportGenerator._parse_markdown_to_flowables(summary_text, styles)
                story.extend(summary_flowables)
            else:
                story.append(PDFReportGenerator._create_paragraph_from_text(
                    "No summary was provided for this query.", styles['Note']
                ))
            
            # Build the document with all content
            doc.build(story, onFirstPage=PDFReportGenerator._first_page_only, onLaterPages=PDFReportGenerator._later_pages)
            
            return True, output_file_path
        
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, str(e)

if __name__ == "__main__":
    # Example usage
    query = "Show me the top 10 customers by revenue"
    sql = "SELECT customer_name, SUM(revenue) as total_revenue FROM sales GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 10"
    
    # Create sample data
    data = {
        'customer_name': [f'Customer {i}' for i in range(1, 11)],
        'total_revenue': [10000*i*(1+0.1*np.random.rand()) for i in range(10, 0, -1)]
    }
    df = pd.DataFrame(data)
    
    summary = """
    # Revenue Analysis Summary
    
    The analysis reveals that **Customer 1** is our highest-value client, generating significant revenue compared to others.
    
    ## Key observations:
    * The top 3 customers contribute over 50% of the total revenue
    * There's a significant drop after Customer 5
    * We should focus retention efforts on the top customers
    
    ### Recommendations
    - Develop personalized retention programs for Customers 1-3
    - Investigate potential growth opportunities with Customers 4-7
    - Consider special promotions for Customers 8-10 to increase their spending
    """
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Generate the PDF report
    output_path = 'output/query_report.pdf'
    success, result = PDFReportGenerator.generate_pdf_report(output_path, query, sql, df, summary)
    
    if success:
        print(f"Report successfully generated at {result}")
    else:
        print(f"Error generating report: {result}")