import pandas as pd
from supabase import create_client, Client

# Your Supabase details
url: str = "https://jlzpxkbuaxnnxjtqahat.supabase.co"  # Replace with your Supabase project URL
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpsenB4a2J1YXhubnhqdHFhaGF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc3NDI3NjYsImV4cCI6MjA2MzMxODc2Nn0._QPhn4tubBbxR3BJRvJubfJJqdxtFbXYUjI4_DKFsJY"   # Replace with your public anon key

supabase: Client = create_client(url, key)

def get_db_connection(db_path: str = None) -> Client:
    return supabase


def get_table_names(conn: Client) -> list[str]:
    # This SQL must not end with a semicolon
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    """
    result = conn.rpc("execute_sql", {"query": query}).execute()
    return [row['table_name'] for row in result.data]


def get_table_schema(conn: Client, table_name: str) -> list[tuple]:
    # This SQL must not end with a semicolon
    query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = '{table_name}'
    """
    data = conn.rpc("execute_sql", {"query": query}).execute()
    return [(row['column_name'], row['data_type']) for row in data.data]


def get_database_schema_string(db_path: str = None) -> str:
    schema_string = "Database Schema:\n"
    try:
        conn = get_db_connection()
        table_names = get_table_names(conn)
        if not table_names:
            return "No tables found in the database."
        for table_name in table_names:
            schema_string += f"Table: {table_name} (\n"
            table_schema = get_table_schema(conn, table_name)
            for col_name, col_type in table_schema:
                schema_string += f"  {col_name} {col_type},\n"
            schema_string = schema_string.rstrip(",\n") + "\n);\n"
    except Exception as e:
        return f"Error loading database schema: {e}"
    return schema_string


def execute_sql_query_db(sql_query: str, db_path: str = None) -> pd.DataFrame | None:
    try:
        # Make sure this query also does NOT end with a semicolon
        if sql_query.strip().endswith(";"):
            sql_query = sql_query.strip().rstrip(";")
        conn = get_db_connection()
        result = conn.rpc("execute_sql", {"query": sql_query}).execute()
        return pd.DataFrame(result.data)
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None


if __name__ == '__main__':
    print("\n--- Database Schema ---")
    schema_str = get_database_schema_string()
    print(schema_str)

    print("\n--- Query Execution Test ---")
    test_query = "SELECT * FROM national_accounts_isic"
    print(f"Executing query: {test_query}")
    result_df = execute_sql_query_db(test_query)

    if result_df is not None:
        print("\nResult DataFrame:")
        print(result_df)
    else:
        print("Query execution failed.")
