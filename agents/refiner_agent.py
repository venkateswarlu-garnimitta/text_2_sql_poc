import os
import time
import openai
import json
import re
import sqlparse # For formatting the output SQL
from dotenv import load_dotenv
load_dotenv()
class RefinerAgent:
    """
    Agent responsible for generating the final SQL query from the decomposition.
    """
    @staticmethod
    def generate_sql(query: str, schema_string: str, decomposition: dict | str= None) -> tuple[str, float]:
        """
        Uses an LLM to generate a SQL query based on the user query, schema, and decomposition.

        Args:
            query (str): The original user's natural language query.
            schema_string (str): A string representation of the database schema.
            decomposition (dict | str): The decomposition of the query (preferably a dictionary).

        Returns:
            tuple[str, float]: A tuple containing:
                                     - str: The generated and formatted SQL query string.
                                     - float: The time taken in seconds.
        """
        start_time = time.time()
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Format decomposition nicely for the prompt if it's a dict
            if decomposition:
                
                decomposition_str = json.dumps(decomposition, indent=2) if isinstance(decomposition, dict) else str(decomposition)
                print(decomposition)
                print(decomposition_str)

                prompt = f"""
                        You are an expert SQL generator.

                        You are given:
                        1. A database schema in SQLite dialect:
                        {schema_string}

                        2. A structured decomposition of the user's natural language query:
                        {decomposition_str}

                        3. The original user query:
                        "{query}"

                        Your task is to generate a valid, optimized SQL query using the provided schema and decomposition that answers the user’s intent accurately.

                        Strict guidelines:
                        - Output only the SQL query as plain text. DO NOT use markdown formatting (no ```sql or ```).
                        - Follow proper SQLite syntax.
                        - Enclose any column or table names that start with a digit or contain special characters in double quotes (e.g., "2023", "user table").
                        - Ensure correct use of GROUP BY for non-aggregated SELECT columns.
                        - Use explicit JOINs if multiple tables are involved.
                        - Avoid subqueries unless necessary for correctness or performance.
                        - Use CTEs (WITH clauses) if required for clarity.
                        - Prioritize readability using aliases and indentation.
                        - The output must be directly runnable in an SQLite engine without modification.

                        Return only the SQL query. No extra explanation or formatting.
                        """
            else:
                prompt = f"""
                You are an expert SQL generator.

                You are given:
                1. A database schema in SQLite dialect:
                {schema_string}

                2. A natural language query from the user:
                "{query}"

                Your task is to generate a valid, optimized SQL query that accurately answers the user’s question using the given schema.

                Strict guidelines:
                - Output only the SQL query as plain text. DO NOT use markdown formatting (no ```sql or ```).
                - Follow proper SQLite syntax.
                - Enclose any column or table names that start with a digit or contain special characters in double quotes (e.g., "2023", "user table").
                - Ensure correct use of GROUP BY for non-aggregated SELECT columns.
                - Use explicit JOINs if multiple tables are involved.
                - Avoid subqueries unless necessary for correctness or performance.
                - Use CTEs (WITH clauses) if required for clarity.
                - Prioritize readability using aliases and indentation.
                - The output must be directly runnable in an SQLite engine without modification.

                Return only the SQL query. No extra explanation or formatting.
                """


            response = client.chat.completions.create(
                model="gpt-4o-mini", # Using gpt-4o-mini as per original code
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            sql_query = response.choices[0].message.content.strip()

            # Attempt to format the SQL for better readability
            try:
                formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            except Exception:
                formatted_sql = sql_query # Fallback to raw if formatting fails

            end_time = time.time()
            time_taken = end_time - start_time

            return formatted_sql, time_taken
        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            error_message = f"Error in RefinerAgent: {str(e)}"
            print(error_message) # Log the error
            return error_message, time_taken

# Example usage (for testing)
if __name__ == '__main__':
    # Dummy schema and decomposition for testing
    dummy_schema = """
    Database Schema:
    Table: employees (
      employee_id INTEGER,
      name TEXT,
      department TEXT,
      salary REAL
    );
    Table: departments (
      department_id INTEGER,
      department_name TEXT
    );
    """
    dummy_decomposition = {
      "tables_needed": ["employees", "departments"],
      "columns_needed": ["employees.salary", "departments.department_name"],
      "aggregations": ["AVG(employees.salary)"],
      "filters_conditions": "None",
      "joins": "employees.department = departments.department_name",
      "grouping": ["departments.department_name"],
      "ordering": ["AVG(employees.salary) DESC"],
      "explanation": "Calculate the average salary for each department by joining employees and departments, group by department name, and order by the calculated average salary descending."
    }
    query = "Show me the average salary for each department, ordered by average salary descending."

    # Ensure OPENAI_API_KEY is set in your environment or a .env file
    if os.getenv("OPENAI_API_KEY"):
        print(f"Generating SQL for query: '{query}'")
        sql, duration = RefinerAgent.generate_sql(query, dummy_schema, dummy_decomposition)
        print("Generated SQL:")
        print(sql)
        print(f"Time taken: {duration:.4f} seconds")
    else:
        print("OpenAI API key not found. Cannot run RefinerAgent example.")
