import os
import time
import openai
import json
import re
class SelectorAgent:
    """
    Agent responsible for checking if a user query is answerable based on the schema.
    """
    @staticmethod
    def is_query_answerable(query: str, schema_string: str) -> tuple[bool, str, float]:
        """
        Uses an LLM to determine if the query can be answered from the provided schema.

        Args:
            query (str): The user's natural language query.
            schema_string (str): A string representation of the database schema.

        Returns:
            tuple[bool, str, float]: A tuple containing:
                                     - bool: True if the query is answerable, False otherwise.
                                     - str: A brief explanation from the LLM.
                                     - float: The time taken in seconds.
        """
        start_time = time.time()
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""
                You are a smart query evaluator.

                Given the following database schema:
                {schema_string}

                And the user query:
                "{query}"

                Your task is to determine two things:

                1. Whether the query is answerable using ONLY the tables and columns in the schema.
                2. Whether this query needs to be decomposed into subqueries based on complexity.

                Decomposition is needed **only if**:
                - The query has multiple parts (e.g., comparisons, trends, combined filters).
                - The query involves multiple tables.
                - The query requires different aggregations or complex logic.
                - The query cannot be answered with a direct simple SELECT using one table.

                If the query is simple, only uses one table, and doesn’t involve multiple operations, then decomposition is **not** needed.

                Return your result as a **JSON object** in the following format:

                {{
                "is_answerable": "Yes" or "No",
                "need_decompose": "Yes" or "No"
                }}

                Examples:
                - {{ "is_answerable": "Yes", "need_decompose": "No" }}
                - {{ "is_answerable": "Yes", "need_decompose": "Yes" }}
                - {{ "is_answerable": "No", "need_decompose": "No" }}

                Important:
                - Only return a **valid JSON object** with these two fields.
                - Do not include any markdown, explanation, or extra text outside the JSON.
                """


            response = client.chat.completions.create(
                model="gpt-4o-mini", # Using gpt-4o-mini as per original code
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50 # Keep the response brief
            )

          

            answer = response.choices[0].message.content.strip()
            print(answer)  # for debugging/logging

            try:
                result = json.loads(answer)

                # Normalize string values to boolean
                is_answerable = result.get("is_answerable", "").strip().lower() == "yes"
                need_decompose = result.get("need_decompose", "").strip().lower() == "yes"
                end_time = time.time()
                time_taken = end_time -start_time
                return is_answerable, need_decompose, time_taken


            except json.JSONDecodeError:
                print(f"Failed to parse JSON from SelectorAgent. Raw response: {answer}")
                return False, False 
            
            

            
        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            error_message = f"Error in SelectorAgent: {str(e)}"
            print(error_message) # Log the error
            return False, error_message, time_taken

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
    query_answerable = "Show me the average salary by department."
    query_not_answerable = "What is the weather like in London?"

    # Ensure OPENAI_API_KEY is set in your environment or a .env file
    if os.getenv("OPENAI_API_KEY"):
        print(f"Checking query: '{query_answerable}'")
        answerable, explanation, duration = SelectorAgent.is_query_answerable(query_answerable, dummy_schema)
        print(f"Answerable: {answerable}, Explanation: {explanation}, Time: {duration:.4f}s")

        print(f"\nChecking query: '{query_not_answerable}'")
        answerable, explanation, duration = SelectorAgent.is_query_answerable(query_not_answerable, dummy_schema)
        print(f"Answerable: {answerable}, Explanation: {explanation}, Time: {duration:.4f}s")
    else:
        print("OpenAI API key not found. Cannot run SelectorAgent example.")
