import os
import time
import openai
import json
import re
from dotenv import load_dotenv
load_dotenv()
class DecomposerAgent:
    """
    Agent responsible for decomposing a complex natural language query.
    """
    @staticmethod
    def decompose_query(query: str, schema_string: str) -> tuple[dict | str, float]:
        """
        Uses an LLM to decompose the user query based on the database schema.

        Args:
            query (str): The user's natural language query.
            schema_string (str): A string representation of the database schema.

        Returns:
            tuple[dict | str, float]: A tuple containing:
                                     - dict | str: The decomposition as a dictionary or an error string.
                                     - float: The time taken in seconds.
        """
        start_time = time.time()
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""
You are an expert SQL planner.

Given the following database schema:
{schema_string}

And the user query:
"{query}"

Decompose this natural language query into structured logical components needed to construct an accurate SQL query.

Please analyze and identify the following fields, and return them in valid JSON format (no markdown, no explanation outside JSON):

1. "tables_needed": List of tables that are relevant for the query. Include any additional tables needed for joins or derived data.

2. "columns_needed": Specific columns required from each table (with table qualifiers if needed). For time-based queries, clarify whether to use:
   - Exact year column(s) (e.g., "2015", "2023")
   - All years in a range
   - A calculated field across years

3. "aggregations": List of aggregation operations required. Each should include:
   - "operation": e.g., SUM, AVG, COUNT, MAX, MIN
   - "target_column": Name of the column(s) involved
   - "alias": Name to use in the output (optional but recommended)

4. "filters_conditions": List of filtering criteria derived from the query. Include:
   - Column(s) involved
   - Operators (e.g., =, >, IN, BETWEEN)
   - Values (exact values, ranges, etc.)

5. "joins": If multiple tables are involved, specify:
   - "left_table", "right_table"
   - "join_type": e.g., INNER JOIN, LEFT JOIN
   - "on": Join condition (e.g., "table1.id = table2.fk_id")

6. "grouping": Columns that should be used in GROUP BY clauses, if any.

7. "ordering": Sort requirements, with:
   - "column"
   - "direction": ASC or DESC

8. "time_handling": Explain how to treat year/time columns if mentioned:
   - "treat_each_year_separately"
   - "sum_across_years"
   - "compare_years"
   - Include "year_range" if applicable (e.g., {{ "start_year": 2015, "end_year": 2023 }})

9. "column_transformation": If year columns need reshaping, specify:
   - "pivot_years_to_rows"
   - "keep_columnar"
   - "aggregate_across_columns"

10. "explanation": A short explanation (1–3 sentences) of what the query is asking for and how it will be answered in SQL.

❗Important rules:
- Quote numeric column names (like "2015") in SQL-style if necessary.
- Always treat time or year ranges with inclusive logic unless explicitly stated.
- If the user says "total", determine if it refers to:
   - Sum across years
   - Sum across categories
   - Both
- Return **only a valid JSON object** — no markdown, comments, or extra text.

Your output should be clean, complete, and syntactically valid JSON that can be parsed directly.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini", # Using gpt-4o-mini as per original code
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            decomposition_text = response.choices[0].message.content.strip()

            # Attempt to parse the JSON response
            try:
                # Remove markdown code block formatting if present
                json_match = re.search(r'```(?:json)?(.*?)```', decomposition_text, re.DOTALL)
                if json_match:
                    decomposition_json_str = json_match.group(1).strip()
                else:
                    decomposition_json_str = decomposition_text

                decomposition = json.loads(decomposition_json_str)
                end_time = time.time()
                time_taken = end_time - start_time
                return decomposition, time_taken
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text and log an error
                end_time = time.time()
                time_taken = end_time - start_time
                error_message = f"DecomposerAgent: Failed to parse JSON response. Raw response: {decomposition_text}"
                print(error_message) # Log the error
                return decomposition_text, time_taken # Return raw text if JSON is invalid

        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            error_message = f"Error in DecomposerAgent: {str(e)}"
            print(error_message) # Log the error
            return error_message, time_taken

# Example usage (for testing)
if __name__ == '__main__':
    # Dummy schema for testing
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
    query = "Show me the average salary for each department, ordered by average salary descending."

    # Ensure OPENAI_API_KEY is set in your environment or a .env file
    if os.getenv("OPENAI_API_KEY"):
        print(f"Decomposing query: '{query}'")
        decomposition, duration = DecomposerAgent.decompose_query(query, dummy_schema)
        print("Decomposition Result:")
        print(json.dumps(decomposition, indent=2))
        print(f"Time taken: {duration:.4f} seconds")
    else:
        print("OpenAI API key not found. Cannot run DecomposerAgent example.")
