import streamlit as st
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path
import re
import base64
import sqlparse # For formatting the output SQL
import numpy as np # Added for table indexing
import io # Added for excel export

# Import agents and database functions
from agents.schema_loader_agent import SchemaLoaderAgent
from agents.selector_agent import SelectorAgent
from agents.decomposer_agent import DecomposerAgent
from agents.refiner_agent import RefinerAgent
from agents.visualization_agent import VisualizationAgent
from database import execute_sql_query_db
from file_uploads import insert_data,create_table_if_not_exists,clean_column_name
from pdf_v3 import PDFReportGenerator
# Load environment variables from .env file
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="NL2SQL Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Prevent overflow */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        overflow-x: hidden !important;
    }

    header {
        visibility: hidden;
    }

    /* Sidebar styling (unchanged width) */
    section[data-testid="stSidebar"] {
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        background-color: #ffffff;
        padding-left: 1rem !important;   /* Optional inner padding */
        margin-left: 0 !important;
    }

    /* Main content area spacing */
    .main, .block-container {
        padding-left: 3rem !important;   /* Space between sidebar and content */
        padding-right: 3rem !important;  /* Space at right edge */
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* Optional: limit content overflow */
    .stApp {
        overflow-x: hidden !important;
    }

    /* Style for agent timing */
    .agent-timing {
        font-size: 0.9em;
        color: #555;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if 'db_schema_string' not in st.session_state:
    st.session_state.db_schema_string = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = ""
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []
if 'selected_visualization' not in st.session_state:
    st.session_state.selected_visualization = "Table"

if "is_database_execution" not in st.session_state:
    st.session_state.is_database_execution = False
if 'insights' not in st.session_state:
    st.session_state.insights = ""
if 'data_insights' not in st.session_state:
    st.session_state.data_insights = ""
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0.0


def create_visualization(df: pd.DataFrame, vis_type: str = "Table"):
    """
    Creates and displays various visualizations using Plotly.
    Insights are now handled in the main app flow within tabs.

    Args:
        df (pd.DataFrame): The data to visualize.
        vis_type (str): The type of visualization to create.
    """
    if df is None or df.empty:
        st.warning("No data available to visualize.")
        return

    graph_width = 800
    graph_height = 550

    def format_aed_currency(value, for_chart_axis_label=False, for_chart_hover=False):
        if pd.isna(value):
            return ""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)

        abs_value = abs(value)
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

        formatted_value = value / divisor

        if for_chart_axis_label:
            return f"{formatted_value:,.1f}{suffix}"
        elif for_chart_hover:
            return f"{value:,.2f} AED"
        else:
            if formatted_value == int(formatted_value):
                return f"{int(formatted_value):,}{suffix} "
            else:
                return f"{formatted_value:,.2f}{suffix} "

    df_processed = df.copy()

    for col in df_processed.select_dtypes(include=['object', 'string']).columns:
        temp_numeric_series = pd.to_numeric(df_processed[col], errors='coerce')
        if temp_numeric_series.notna().all():
            df_processed[col] = temp_numeric_series
        else:
            contains_currency_symbol = False
            for val in df_processed[col].dropna().head(10):
                if isinstance(val, str) and ('AED' in val or '€' in val or '$' in val or '£' in val or '₹' in val):
                    contains_currency_symbol = True
                    break
            if contains_currency_symbol:
                df_processed[col] = df_processed[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    numeric_cols_current = df_processed.select_dtypes(include=['number']).columns
    identified_currency_cols = set()

    keyword_currency_cols = [
        col for col in numeric_cols_current
        if any(keyword in col.lower() for keyword in ['amount', 'value', 'price', 'total', 'revenue', 'cost', 'spend', 'budget', 'sales', 'profit', 'aed'])
    ]
    identified_currency_cols.update(keyword_currency_cols)

    numeric_suffix_pattern = re.compile(r'_\d+$')
    suffix_currency_cols = [
        col for col in numeric_cols_current
        if numeric_suffix_pattern.search(col)
    ]
    identified_currency_cols.update(suffix_currency_cols)

    LARGE_VALUE_THRESHOLD = 1000
    for col in numeric_cols_current:
        if col not in identified_currency_cols:
            if not df_processed[col].empty and df_processed[col].count() > 0:
                non_na_values = df_processed[col].dropna()
                if not non_na_values.empty:
                    is_integer_series = np.isclose(non_na_values, non_na_values.astype(int))
                    if is_integer_series.all():
                        count_large_integers = (non_na_values >= LARGE_VALUE_THRESHOLD).sum()
                        percentage_large_integers = count_large_integers / len(non_na_values)
                        if percentage_large_integers > 0.5:
                            identified_currency_cols.add(col)
    currency_like_cols = list(identified_currency_cols)

    df_display = df_processed.copy()
    for col in numeric_cols_current:
        if col in currency_like_cols:
            df_display[col] = df_display[col].apply(lambda x: format_aed_currency(x, for_chart_axis_label=False, for_chart_hover=False))
        else:
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and x != int(x) else (f"{int(x):,}" if isinstance(x, (int, float)) else x))

    # Common font settings for bold black labels (applied where possible)
    label_font_settings = dict(family="Arial, sans-serif", size=12, color="black") # General label font
    title_font_settings = dict(family="Arial Black, sans-serif", size=16, color="black") # Title font (bolder)


    if vis_type == "Table":
        rename_map = {col: f"{col} (AED)" for col in currency_like_cols}
        df_display_renamed = df_display.rename(columns=rename_map)
        # Start index from 1 for display
        df_display_renamed.index = df_display_renamed.index + 1
        st.dataframe(df_display_renamed, use_container_width=True)

    elif vis_type == "Bar Chart":
        categorical_cols = df_processed.select_dtypes(exclude=['number', 'datetime64[ns]']).columns
        if len(numeric_cols_current) > 0 and len(categorical_cols) > 0:
            x_col = st.selectbox("Select X-axis (categorical):", categorical_cols)
            y_col = st.selectbox("Select Y-axis (numeric):", numeric_cols_current)

            if x_col and y_col:
                grouped_data = df_processed.groupby(x_col)[y_col].sum().reset_index()
                fig = px.bar(
                    grouped_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}",
                    width=graph_width, height=graph_height, color=x_col,
                    color_discrete_sequence=px.colors.qualitative.Bold, text_auto=False
                )
                y_axis_title = f"{y_col} (AED)" if y_col in currency_like_cols else y_col
                fig.update_layout(
    title=dict(
        text=f"{y_col} by {x_col}",
        font=dict(color="black", size=18)
    ),
    font=dict(color="black", size=12),  # Applies to ticks, legend
    xaxis=dict(
        title=dict(text=x_col, font=dict(color="black", size=14)),
        tickfont=dict(color="black"),
    ),
    yaxis=dict(
        title=dict(text=y_axis_title, font=dict(color="black", size=14)),
        tickfont=dict(color="black"),
    ),
    legend=dict(
        font=dict(color="black", size=12)
    ),
    hoverlabel=dict(
        font=dict(color="black")
    ),
    template="plotly_white",
    plot_bgcolor='rgba(240,240,250,0.2)',
    paper_bgcolor='white',
    bargap=0.2,
    bargroupgap=0.1,
    hovermode="x unified",
    title_x=0.5
)

                if y_col in currency_like_cols:
                    fig.update_yaxes(tickformat=".2s", hoverformat=",.2f", gridcolor='rgba(200,200,200,0.2)', title=y_axis_title, tickfont=dict(color="black"))
                    fig.update_traces(
                        text=[format_aed_currency(val, for_chart_axis_label=True) for val in grouped_data[y_col]],
                        textposition='outside', textfont=dict(size=10, color='black'),
                        customdata=[format_aed_currency(x, for_chart_hover=True) for x in grouped_data[y_col]],
                        hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{customdata}}<extra></extra>"
                    )
                else:
                    fig.update_yaxes(tickformat=",.2f", gridcolor='rgba(200,200,200,0.2)', tickfont=dict(color="black"))
                    fig.update_traces(
                        text=[f"{val:,.1f}" for val in grouped_data[y_col]],
                        textposition='outside', textfont=dict(size=10, color='black'),
                        hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:,.2f}}<extra></extra>"
                    )
                fig.update_xaxes(tickfont=dict(color="black"))
                st.plotly_chart(fig)
            else:
                st.error("Please select both X and Y columns.")
        else:
            st.error("Bar chart requires at least one categorical and one numeric column.")

    elif vis_type == "Line Chart":
        datetime_cols = df_processed.select_dtypes(include=['datetime64[ns]']).columns
        potential_x_cols = list(datetime_cols) + [col for col in df_processed.columns if col not in datetime_cols and col not in numeric_cols_current]
        if len(numeric_cols_current) >= 1 and len(potential_x_cols) >= 1:
            x_col = st.selectbox("Select X-axis:", potential_x_cols)
            y_cols = st.multiselect("Select Y-axis (numeric):", numeric_cols_current)
            if x_col and y_cols:
                x_data = df_processed[x_col]
                if df_processed[x_col].dtype == 'object':
                    try:
                        x_data = pd.to_datetime(df_processed[x_col])
                    except Exception:
                        st.warning(f"Could not convert '{x_col}' to datetime. Using as-is.")
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, y_col_single in enumerate(y_cols):
                    current_customdata = None
                    if y_col_single in currency_like_cols:
                        current_customdata = df_processed[y_col_single].apply(lambda x: format_aed_currency(x, for_chart_hover=True)).values
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df_processed[y_col_single], mode='lines+markers', name=y_col_single,
                        line=dict(width=3, color=colors[i % len(colors)], shape='spline'),
                        marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
                        customdata=current_customdata
                    ))
                y_axis_title = "Value (AED)" if any(col in currency_like_cols for col in y_cols) else "Value"
                fig.update_layout(
                    title_text=f"Line Chart: {', '.join(y_cols)} over {x_col}", title_font=title_font_settings,
                    xaxis_title=x_col, yaxis_title=y_axis_title,
                    width=graph_width, height=graph_height,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.5)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
                    hovermode="x unified", template="plotly_white", title_x=0.5,
                    font=label_font_settings,
                    xaxis_title_font=dict(color="black", size=14),
                    yaxis_title_font=dict(color="black", size=14),
                    plot_bgcolor='rgba(240,240,250,0.2)', paper_bgcolor='white'
                )
                if any(col in currency_like_cols for col in y_cols):
                    fig.update_yaxes(tickformat=".2s", hoverformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, title=y_axis_title, tickfont=dict(color="black"))
                else:
                    fig.update_yaxes(tickformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, tickfont=dict(color="black"))
                
                fig.update_xaxes(tickfont=dict(color="black"))

                for i, trace in enumerate(fig.data):
                    if y_cols[i] in currency_like_cols:
                        trace.hovertemplate = f"<b>%{{x}}</b><br>{y_cols[i]}: %{{customdata}}<extra></extra>"
                    else:
                        trace.hovertemplate = f"<b>%{{x}}</b><br>{y_cols[i]}: %{{y:,.2f}}<extra></extra>"
                st.plotly_chart(fig)
            else:
                st.error("Please select X and at least one Y column.")
        else:
            st.error("Line chart requires at least one suitable X-axis column (numeric/datetime) and one numeric Y-axis column.")

    elif vis_type == "Scatter Plot":
        if len(numeric_cols_current) >= 2:
            x_col = st.selectbox("Select X-axis:", numeric_cols_current)
            default_y_index = 0 if len(numeric_cols_current) < 2 or numeric_cols_current[0] != x_col else 1
            y_col = st.selectbox("Select Y-axis:", numeric_cols_current, index=min(default_y_index, len(numeric_cols_current)-1))
            color_option = None
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                use_color = st.checkbox("Use color grouping?")
                if use_color:
                    color_option = st.selectbox("Select color column:", categorical_cols)
            if x_col and y_col:
                fig = px.scatter(
                    df_processed, x=x_col, y=y_col, color=color_option,
                    title=f"Scatter Plot: {y_col} vs {x_col}",
                    width=graph_width, height=graph_height, template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    hover_name=df_processed.index.name if df_processed.index.name else None, opacity=0.8
                )
                x_axis_title = f"{x_col} (AED)" if x_col in currency_like_cols else x_col
                y_axis_title = f"{y_col} (AED)" if y_col in currency_like_cols else y_col
                fig.update_layout(
                    title_font=title_font_settings,
                    xaxis_title=x_axis_title, yaxis_title=y_axis_title,
                    hovermode="closest", title_x=0.5,
                    font=label_font_settings,
                    xaxis_title_font=dict(color="black", size=14),
                    yaxis_title_font=dict(color="black", size=14),
                    plot_bgcolor='rgba(240,240,250,0.2)', paper_bgcolor='white',
                    legend=dict(bgcolor='rgba(255,255,255,0.5)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1) if color_option else None
                )
                if x_col in currency_like_cols:
                    fig.update_xaxes(tickformat=".2s", hoverformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, title=x_axis_title, tickfont=dict(color="black"))
                else:
                    fig.update_xaxes(tickformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, tickfont=dict(color="black"))
                if y_col in currency_like_cols:
                    fig.update_yaxes(tickformat=".2s", hoverformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, title=y_axis_title, tickfont=dict(color="black"))
                else:
                    fig.update_yaxes(tickformat=",.2f", gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)', zerolinewidth=1, tickfont=dict(color="black"))
                
                custom_hover_data = pd.DataFrame(index=df_processed.index)
                x_hover_template = f"<b>{x_col}</b>: %{{customdata[0]}}" if x_col in currency_like_cols else f"<b>{x_col}</b>: %{{x:,.2f}}"
                if x_col in currency_like_cols:
                    custom_hover_data[x_col] = df_processed[x_col].apply(lambda val: format_aed_currency(val, for_chart_hover=True))

                y_hover_template = f"<b>{y_col}</b>: %{{customdata[1 if x_col in currency_like_cols else 0]}}" if y_col in currency_like_cols else f"<b>{y_col}</b>: %{{y:,.2f}}"
                if y_col in currency_like_cols:
                     custom_hover_data[y_col] = df_processed[y_col].apply(lambda val: format_aed_currency(val, for_chart_hover=True))


                fig.update_traces(
                    marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
                    customdata=custom_hover_data.values,
                    hovertemplate=f"{x_hover_template}<br>{y_hover_template}<extra></extra>"
                )
                st.plotly_chart(fig)
            else:
                st.error("Please select both X and Y columns.")
        else:
            st.error("Scatter plot requires at least two numeric columns.")

    elif vis_type == "Pie Chart":
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and len(numeric_cols_current) > 0:
            label_col = st.selectbox("Select label column:", categorical_cols)
            value_col = st.selectbox("Select value column:", numeric_cols_current)
            if label_col and value_col:
                grouped_data = df_processed.groupby(label_col)[value_col].sum().reset_index()
                if grouped_data[label_col].nunique() > 10:
                    grouped_data = grouped_data.sort_values(by=value_col, ascending=False)
                    top_values = grouped_data.head(10)
                    other_sum = grouped_data.iloc[10:][value_col].sum()
                    if other_sum > 0:
                        other_row = pd.DataFrame({label_col: ['Other'], value_col: [other_sum]})
                        grouped_data = pd.concat([top_values, other_row], ignore_index=True)
                    else:
                        grouped_data = top_values
                    st.info("Showing only top 10 categories, others are grouped.")
                total = grouped_data[value_col].sum()
                fig = px.pie(
                    grouped_data, names=label_col, values=value_col,
                    title=f"Distribution of {value_col} by {label_col}",
                    width=graph_width, height=graph_height,
                    color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4
                )
                hovertemplate_currency = f"<b>%{{label}}</b><br>{value_col}: %{{value:,.2f}} AED<br>Percent: %{{percent:.1%}}<extra></extra>"
                hovertemplate_numeric = f"<b>%{{label}}</b><br>{value_col}: %{{value:,.2f}}<br>Percent: %{{percent:.1%}}<extra></extra>"
                fig.update_traces(
                    textposition='outside', textinfo='label+percent',
                    marker=dict(line=dict(color='#FFFFFF', width=2)),
                    pull=[0.05 if x == grouped_data[value_col].idxmax() else 0 for x in grouped_data.index],
                    hovertemplate=hovertemplate_currency if value_col in currency_like_cols else hovertemplate_numeric,
                    rotation=90,
                    textfont=dict(color="black") # Make pie slice labels black
                )
                fig.update_layout(
                    title_font=title_font_settings,
                    uniformtext_minsize=12, uniformtext_mode='hide', title_x=0.5,
                    font=label_font_settings, # General font for legend etc.
                    paper_bgcolor='white',
                    legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(color="black")),
                    annotations=[dict(
                        text=f"Total:<br>{format_aed_currency(total, for_chart_axis_label=True) + ' AED' if value_col in currency_like_cols else f'{total:,.2f}'}",
                        showarrow=False, font=dict(size=14, color="black"), x=0.5, y=0.5 # Center text black
                    )]
                )
                st.plotly_chart(fig)
            else:
                st.error("Please select both label and value columns.")
        else:
            st.error("Pie chart requires at least one categorical and one numeric column.")

    elif vis_type == "Heatmap":
        if len(numeric_cols_current) >= 2:
            default_cols_for_heatmap = list(numeric_cols_current[:min(10, len(numeric_cols_current))])
            selected_cols = st.multiselect("Select columns for correlation heatmap:", numeric_cols_current, default=default_cols_for_heatmap)
            if selected_cols:
                df_selected_numeric = df_processed[selected_cols].select_dtypes(include=['number'])
                if not df_selected_numeric.empty:
                    correlation = df_selected_numeric.corr()
                    labels = [f"{col} (AED)" if col in currency_like_cols else col for col in correlation.columns]
                    fig = px.imshow(
                        correlation, text_auto='.2f', color_continuous_scale='RdBu_r', aspect="auto",
                        title="Correlation Heatmap", width=graph_width + 100, height=graph_height + 100,
                        x=labels, y=labels, zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        title_font=title_font_settings,
                        xaxis_title="", yaxis_title="", xaxis_tickangle=-45, title_x=0.5,
                        font=label_font_settings, # General font for color bar labels etc.
                        paper_bgcolor='white',
                        xaxis_tickfont=dict(color="black", size=10), # Tick labels black
                        yaxis_tickfont=dict(color="black", size=10)  # Tick labels black
                    )
                    fig.update_traces(
                        hovertemplate='<b>Correlation</b><br>X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>',
                        texttemplate='%{z:.2f}',
                        textfont=dict(color="black") # Text on heatmap cells
                    )
                    st.plotly_chart(fig)
                else:
                    st.error("No numeric columns selected for heatmap.")
            else:
                st.info("Select at least two numeric columns for the heatmap.")
        else:
            st.error("Heatmap requires at least two numeric columns.")
    if st.session_state.insights:
        st.markdown("---")
        st.markdown("### 💡 Summary Insights")
        st.markdown(st.session_state.insights, unsafe_allow_html=False) # Set to False if it's plain text
               

    # Removed insight display from here, will be handled in main app tabs

def process_query(query: str):
    """
    Processes the user's natural language query through the agent pipeline.
    """
    start_total_time = time.time()
    if not st.session_state.db_schema_string:
        st.error("Database schema not loaded. Please ensure 'data/my_database.db' exists and is accessible.")
        return 0.0

    st.session_state.current_query = query
    st.session_state.processing_steps = []
    st.session_state.result_df = None
    st.session_state.sql_query = ""
    st.session_state.insights = ""
    st.session_state.data_insights = "" # Clear data insights as well
    st.session_state.is_database_execution = False # Reset flag

    if query and query not in st.session_state.query_history:
        st.session_state.query_history.append(query)
        if len(st.session_state.query_history) > 10:
            st.session_state.query_history.pop(0)

    if st.session_state.db_schema_string:
        st.session_state.processing_steps.append({
            "agent": "Schema Loader", "status": "✅ Schema already loaded from database",
            "details": "Database schema is available.", "time_taken": 0.0
        })
    else:
        st.session_state.processing_steps.append({
            "agent": "Schema Loader", "status": "❌ Schema not loaded",
            "details": "Database schema could not be loaded.", "time_taken": 0.0
        })
        st.error("Database schema is not available. Cannot process query.")
        return 0.0

    with st.spinner("Selector Agent: Checking if query is answerable..."):
        is_answerable, need_decompose, time_taken = SelectorAgent.is_query_answerable(
            query, st.session_state.db_schema_string
        )
    status = "✅ Query is answerable" if is_answerable else "❌ Query is not answerable"
    st.session_state.processing_steps.append({
        "agent": "Selector Agent", "status": status,
        "details": f"Need decomposer: {need_decompose}", "time_taken": time_taken
    })
    if not is_answerable:
        st.warning("Query cannot be answered with the current database schema.")
        st.session_state.total_time = time.time() - start_total_time
        return st.session_state.total_time


    decomposition = None
    if need_decompose:
        with st.spinner("Decomposer Agent: Breaking down query..."):
            decomposition, time_taken = DecomposerAgent.decompose_query(
                query, st.session_state.db_schema_string
            )
        st.session_state.processing_steps.append({
            "agent": "Decomposer Agent", "status": "✅ Query decomposed",
            "details": decomposition, "time_taken": time_taken
        })

    with st.spinner("Refiner Agent: Generating SQL..."):
        sql_query, time_taken = RefinerAgent.generate_sql(
            query, st.session_state.db_schema_string, decomposition
        )
    st.session_state.sql_query = sql_query
    st.session_state.processing_steps.append({
        "agent": "Refiner Agent", "status": "✅ SQL generated",
        "details": sql_query, "time_taken": time_taken
    })

    with st.spinner("Executing SQL query..."):
        start_time_exec = time.time()
        result_df = execute_sql_query_db(sql_query)
        time_taken_exec = time.time() - start_time_exec
    st.session_state.result_df = result_df
    st.session_state.is_database_execution = True # Set flag after execution attempt

    if result_df is not None:
        st.session_state.processing_steps.append({
            "agent": "Database Execution", "status": "✅ Query executed successfully",
            "details": f"Query returned {len(result_df)} rows and {len(result_df.columns)} columns.",
            "time_taken": time_taken_exec
        })
    else:
        st.session_state.processing_steps.append({
            "agent": "Database Execution", "status": "❌ Query execution failed",
            "details": "The generated SQL query could not be executed.",
            "time_taken": time_taken_exec
        })
        st.error("SQL query execution failed. Please check the generated SQL.")
        st.session_state.total_time = time.time() - start_total_time
        return st.session_state.total_time


    if result_df is not None and not result_df.empty:
        with st.spinner("Visualization Agent: Suggesting visualization & generating insights..."):
            vis_type, time_taken, insights_text, data_insights_text = VisualizationAgent.suggest_visualization(query, result_df) # Ensure it returns two distinct insights
        st.session_state.selected_visualization = vis_type
        
        insights_text = insights_text.strip().strip('```json').strip('```').strip().strip('}').strip().strip(']').strip().strip(': [')
        st.session_state.insights = insights_text.encode().decode("unicode_escape")
        
        data_insights_text = data_insights_text.strip().strip('```json').strip('```').strip().strip('}').strip().strip(']').strip().strip(': [')
        st.session_state.data_insights = data_insights_text.encode().decode("unicode_escape")

        st.session_state.processing_steps.append({
            "agent": "Visualization Agent", "status": "✅ Visualization suggested with insights",
            "details": f"**Suggested visualization:** `{vis_type}`\n\n**Summary & Data Insights generated.**",
            "time_taken": time_taken
        })
    elif result_df is not None and result_df.empty:
        st.info("Query executed successfully, but returned no data.")
        st.session_state.processing_steps.append({
            "agent": "Visualization Agent", "status": "ℹ️ No data for visualization",
            "details": "Query returned an empty result set.", "time_taken": 0.0
        })
    
    st.session_state.total_time = time.time() - start_total_time
    return st.session_state.total_time

def load_schema_on_startup():
    if st.session_state.db_schema_string is None or "Error loading database schema" in st.session_state.db_schema_string:
        with st.spinner(f"Loading database schema..."):
            schema_string, time_taken = SchemaLoaderAgent.load_schema_from_db()
            st.session_state.db_schema_string = schema_string
        if st.session_state.db_schema_string and "Error loading database schema" not in st.session_state.db_schema_string:
            st.sidebar.success("Database schema loaded.")
        else:
            st.sidebar.error(f"Failed to load database schema. Ensure the DB file exists and is valid.")


def main():
    load_schema_on_startup()

    with st.sidebar:
        st.title("📊 NL2SQL Assistant")
        st.subheader("📝 About")
        st.markdown("""
        This application converts natural language to SQL queries for a SQLite database using AI agents:
        - **Schema Loader**: Extracts database structure.
        - **Selector Agent**: Validates query answerable from schema.
        - **Decomposer Agent**: Breaks complex queries into logical parts.
        - **Refiner Agent**: Converts to clean SQL for SQLite.
        - **Visualization Agent**: Suggests appropriate visualizations and generates insights.
        """)
        st.subheader("Database Status")
        if st.session_state.db_schema_string and "Error loading database schema" not in st.session_state.db_schema_string:
            st.success("Database schema loaded.")
            with st.expander("View Schema"):
                st.code(st.session_state.db_schema_string, language="text")
        elif st.session_state.db_schema_string:
            st.error(st.session_state.db_schema_string)
        else:
            st.warning("Database schema not loaded.")

        uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("🚀 Create Tables and Insert Data"):
                all_success = True
                for uploaded_file in uploaded_files:
                    table_name = clean_column_name(uploaded_file.name.replace(".csv", ""))
                    st.markdown(f"### Processing: `{uploaded_file.name}` as table `{table_name}`")
                    try:
                        df = pd.read_csv(uploaded_file)
                        # st.dataframe(df.head()) # Optionally show head
                        if create_table_if_not_exists(table_name, df):
                            insert_data(df, table_name)
                            st.success(f"Table `{table_name}` created/updated.")
                        else:
                            st.warning(f"Table `{table_name}` might not have been created as expected.")
                    except Exception as e:
                        all_success = False
                        st.error(f"❌ Failed to process `{uploaded_file.name}`: {e}")
                if all_success:
                    st.success("✅ All files processed successfully!")
                    st.session_state.db_schema_string = None
                    st.rerun()
                else:
                    st.error("Some files could not be processed. Check errors above.")
            # Removed redundant "Please upload" message here

        if st.session_state.query_history:
            st.subheader("📜 Query History")
            for historic_query in reversed(st.session_state.query_history):
                if st.button(f"📝 {historic_query[:40]}...", key=f"hist_{hash(historic_query)}"):
                    # Set the text area value and then process
                    st.session_state.query_text_area = historic_query # Assuming you use this key for text_area
                    process_query(historic_query) # Directly call process
                    st.rerun() # Rerun to update displays immediately


    st.title("🔍 Natural Language to SQL")
    
    # Use a key for the text_area to allow updating it from history
    query = st.text_area("Enter your query in natural language:",
                         value=st.session_state.get('query_text_area', ''), # Use session state for value
                         placeholder="Example: Show me the number of employees in each department.", height=100, key="query_input_main")
    
    # Update session state if user types directly
    st.session_state.query_text_area = query


    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Process Query", type="primary", disabled=not query or not st.session_state.db_schema_string or ("Error" in st.session_state.db_schema_string if st.session_state.db_schema_string else True)):
            total_time = process_query(query)
            st.session_state.total_time = total_time
            # No need to print total_time here unless for debugging
    with col2:
        if st.button("Clear Results", disabled=st.session_state.result_df is None and not st.session_state.processing_steps):
            st.session_state.result_df = None
            st.session_state.sql_query = ""
            st.session_state.processing_steps = []
            st.session_state.current_query = ""
            st.session_state.selected_visualization = "Table"
            st.session_state.insights = ""
            st.session_state.data_insights = ""
            st.session_state.query_text_area = "" # Clear text area
            st.session_state.total_time = 0.0
            st.rerun()

    if st.session_state.processing_steps:
        if st.session_state.total_time > 0:
            st.subheader(f"🔄 Processing Pipeline (Total time: {st.session_state.total_time:.2f}s)")
        else:
            st.subheader("🔄 Processing Pipeline")

        tab_titles = [
            f"{step['agent']} ({step['time_taken']:.2f}s)" if step.get('time_taken') is not None else step['agent']
            for step in st.session_state.processing_steps
        ]
        tabs = st.tabs(tab_titles)

        for i, (tab, step) in enumerate(zip(tabs, st.session_state.processing_steps)):
            with tab:
                st.markdown(f"**Status**: {step['status']}")
                st.write("**Details**:")

                if step["agent"] == "Decomposer Agent":
                    try:
                        if isinstance(step["details"], dict):
                            st.json(step["details"])
                        else:
                            json_match = re.search(r'```(?:json)?(.*?)```', str(step["details"]), re.DOTALL)
                            if json_match:
                                try:
                                    json_data = json.loads(json_match.group(1).strip())
                                    st.json(json_data)
                                except json.JSONDecodeError:
                                    st.code(step["details"], language="text")
                            else:
                                st.code(step["details"], language="text")
                    except Exception:
                        st.code(step["details"], language="text")
                elif step["agent"] == "Refiner Agent":
                    st.code(step["details"], language="sql")
                elif step["agent"] == "Database Execution":
                    st.write(step["details"])
                    if st.session_state.sql_query:
                        st.write("**Executed SQL:**")
                        st.code(st.session_state.sql_query, language="sql")
                    # Display Data Insights in Database Execution tab
                    if st.session_state.data_insights:
                        st.markdown("---")
                        st.markdown("### 📊 Data Insights")
                        st.markdown(st.session_state.data_insights, unsafe_allow_html=False) # Set to False if it's plain text
                elif step["agent"] == "Visualization Agent":
                    st.write(step["details"])
                    # Display General Insights in Visualization Agent tab
                    if st.session_state.insights:
                        st.markdown("---")
                        st.markdown("### 💡 Summary Insights")
                        st.markdown(st.session_state.insights, unsafe_allow_html=False) # Set to False if it's plain text
                else: # Schema Loader, Selector Agent
                    st.write(step["details"])


    if st.session_state.sql_query:
        st.subheader("🔍 Generated SQL Query")
        try:
            formatted_sql_display = sqlparse.format(st.session_state.sql_query, reindent=True, keyword_case='upper')
        except Exception:
            formatted_sql_display = st.session_state.sql_query
        st.code(formatted_sql_display, language="sql")

    if st.session_state.result_df is not None:
        st.subheader("📊 Query Results")
        col_rows, col_cols, col_vis = st.columns([1, 1, 2])
        with col_rows:
            st.metric("Rows", len(st.session_state.result_df))
        with col_cols:
            st.metric("Columns", len(st.session_state.result_df.columns))
        with col_vis:
            visualization_options = ["Table", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap"]
            try:
                default_vis_index = visualization_options.index(st.session_state.selected_visualization)
            except ValueError:
                default_vis_index = 0
            selected_vis = st.selectbox("Visualization", visualization_options, index=default_vis_index)
            st.session_state.selected_visualization = selected_vis

        create_visualization(st.session_state.result_df, st.session_state.selected_visualization)

        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = PDFReportGenerator.generate_report(
                    query=st.session_state.current_query, # Use current_query
                    sql_query = st.session_state.sql_query,
                    result_df = st.session_state.result_df,
                    summary = st.session_state.insights, # Use general insights for PDF summary
                    output_path=None
                )
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="query_report.pdf">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)

        st.subheader("💾 Export Results")
        col_csv, col_excel = st.columns(2)
        with col_csv:
            # Make sure index starts from 1 for CSV export as well if desired, or keep default (False)
            df_export_csv = st.session_state.result_df.copy()
            df_export_csv.index = df_export_csv.index + 1 # If 1-based index needed in CSV
            csv_data = df_export_csv.to_csv().encode('utf-8') # Include index by removing index=False
            st.download_button(
                label="Download as CSV (with index)", data=csv_data,
                file_name=f"query_result_{int(time.time())}.csv", mime="text/csv",
            )
        with col_excel:
            output = io.BytesIO() # Use io.BytesIO for in-memory Excel file
            df_export_excel = st.session_state.result_df.copy()
            df_export_excel.index = df_export_excel.index + 1 # If 1-based index needed in Excel
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                 df_export_excel.to_excel(writer, sheet_name='Sheet1') # Include index by removing index=False
            excel_data = output.getvalue()
            st.download_button(
                label="Download as Excel (with index)", data=excel_data,
                file_name=f"query_result_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    # Ensure output directory exists for temp files if any agent uses it, though excel export is now in-memory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    main()