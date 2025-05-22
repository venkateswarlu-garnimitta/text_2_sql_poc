import streamlit as st
import pandas as pd
from supabase import create_client, Client
import re

# Supabase config
url: str = "https://jlzpxkbuaxnnxjtqahat.supabase.co"  # Replace with your Supabase project URL
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpsenB4a2J1YXhubnhqdHFhaGF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc3NDI3NjYsImV4cCI6MjA2MzMxODc2Nn0._QPhn4tubBbxR3BJRvJubfJJqdxtFbXYUjI4_DKFsJY"   # Replace with your public anon key

supabase: Client = create_client(url, key)


# Clean column names to be SQL-safe
def clean_column_name(col: str) -> str:
    cleaned = re.sub(r"\W+", "_", col.strip().lower())
    if re.match(r"^\d", cleaned):
        cleaned = "_" + cleaned
    return cleaned

# Map pandas dtypes to SQL types
def map_dtype(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"

# Generate CREATE TABLE SQL
def create_table_sql(table_name: str, df: pd.DataFrame) -> str:
    cols = []
    for col in df.columns:
        col_name = clean_column_name(col)
        sql_type = map_dtype(df[col].dtype)
        cols.append(f'"{col_name}" {sql_type}')  # Column names wrapped in quotes
    return f'CREATE TABLE "{table_name}" ({", ".join(cols)});'

# Run a raw SQL query
def run_sql(sql: str):
    return supabase.rpc("run_sql", {"query": sql}).execute()

# Check if table exists
def table_exists(table_name: str) -> bool:
    sql = f"SELECT to_regclass('public.\"{table_name}\"') AS exists;"
    result = supabase.rpc("run_sql", {"query": sql}).execute()
    print(f"Table exists check result: {result.data}")

    if result.data and result.data[0].get("exists") is not None:
        return True
    return False



# Create table safely
def create_table_if_not_exists(table_name: str, df: pd.DataFrame) -> bool:
    if table_exists(table_name):
        st.warning(f"⚠️ Table `{table_name}` already exists. Skipping creation and insertion.")
        return False
    try:
        sql = create_table_sql(table_name, df)
        result = run_sql(sql)
        if result.data is not None:
            st.success(f"✅ Table `{table_name}` created successfully.")
            return True
        
    except Exception as e:
        st.error(f"❌ SQL Execution Error (creating {table_name}): {e}")
        return False

# Insert data into Supabase
# Insert data into Supabase with checks
def insert_data(df: pd.DataFrame, table_name: str):
    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]
    data = df.to_dict(orient="records")

    # Fetch actual columns from Supabase table
    try:
        response = supabase.table(table_name).select("*").limit(1).execute()
        if not response.data:
            st.warning(f"⚠️ Table `{table_name}` is empty. Proceeding with insert.")
        else:
            table_columns = set(response.data[0].keys())
            df_columns = set(df.columns)
            if not df_columns.issubset(table_columns):
                st.error(f"❌ Column mismatch in `{table_name}`.\n\nExpected: {table_columns}\n\nProvided: {df_columns}")
                return
    except Exception as e:
        st.error(f"❌ Failed to fetch columns for `{table_name}`: {e}")
        return

    try:
        result = supabase.table(table_name).insert(data).execute()
        if result.data:
            st.success(f"✅ Inserted {len(result.data)} rows into `{table_name}`.")
        else:
            st.warning(f"⚠️ Insert attempt complete but no data returned for `{table_name}`.")
    except Exception as e:
        st.error(f"❌ Insert Error for `{table_name}`: {e}")


# Streamlit UI
# st.title("📥 Upload CSVs → Supabase Tables")

# uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# if uploaded_files:
#     if st.button("🚀 Create Tables and Insert Data"):
#         for uploaded_file in uploaded_files:
#             table_name = clean_column_name(uploaded_file.name.replace(".csv", ""))
#             st.markdown(f"### Processing: `{uploaded_file.name}` as table `{table_name}`")

#             try:
#                 df = pd.read_csv(uploaded_file)
#                 st.dataframe(df.head())

#                 if create_table_if_not_exists(table_name, df):
#                     insert_data(df, table_name)

#             except Exception as e:
#                 st.error(f"❌ Failed to process `{uploaded_file.name}`: {e}")
# else:
#     st.info("⬆️ Please upload one or more CSV files to get started.")
