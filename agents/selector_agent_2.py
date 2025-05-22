import os
import time
import openai
import json
from collections import deque
from dotenv import load_dotenv
load_dotenv()
class SelectorAgent:
    """
    Selector Agent that checks if a query is answerable, needs decomposition,
    and reframes it in natural language using schema and past context.
    """
    def __init__(self, max_memory=10):
        self.memory = deque(maxlen=max_memory)  # Stores (original_query, reframed_query)

    def add_to_memory(self, original_query, reframed_query):
        self.memory.append((original_query, reframed_query))

    def get_memory_context(self):
        if not self.memory:
            return ""
        context = "Past interactions:\n"
        for i, (query, reframed) in enumerate(reversed(self.memory), 1):
            context += f"{i}. User asked: \"{query}\"\n   Reframed as: \"{reframed}\"\n"
        return context

    def is_query_answerable(self, query: str, schema_string: str) -> tuple[bool, bool, str, float]:
        """
        Returns whether the query is answerable, needs decomposition, and a refined natural language query.
        """
        start_time = time.time()

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            memory_context = self.get_memory_context()

            prompt = f"""
You are an assistant that helps users reframe their natural language questions more clearly using a database schema and previous questions.

{memory_context}

Here is the database schema:
{schema_string}

User's new question:
"{query}"

Tasks:
1. Decide if the question can be answered using the schema.
2. Decide if the question is complex and needs to be broken into smaller sub-questions.
3. Reframe the question clearly using table and column names from the schema and past interactions. Keep it in natural language — do not convert to SQL.

Return this JSON format:
{{
  "is_answerable": "Yes" or "No",
  "need_decompose": "Yes" or "No",
  "reframed_query": "Your reframed question in natural language"
}}

Only return the JSON. Do not include any explanation.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )

            answer = response.choices[0].message.content.strip()
            print("Raw response:", answer)  # Debug print

            try:
                result = json.loads(answer)

                is_answerable = result.get("is_answerable", "").strip().lower() == "yes"
                need_decompose = result.get("need_decompose", "").strip().lower() == "yes"
                reframed_query = result.get("reframed_query", query)

                # Store in memory
                self.add_to_memory(query, reframed_query)

                return is_answerable, need_decompose, reframed_query, time.time() - start_time

            except json.JSONDecodeError:
                print("JSON parsing failed. Raw content:", answer)
                return False, False, "Parsing failed", time.time() - start_time

        except Exception as e:
            return False, False, f"Error: {str(e)}", time.time() - start_time
if __name__ == '__main__':
    schema = """
    Table: nationals (
      id INTEGER,
      sector TEXT,
      score INTEGER
    );
    Table: employees (
      id INTEGER,
      department TEXT,
      salary FLOAT
    );
    """

    agent = SelectorAgent()

    test_queries = [
        "Show top 5 rows",                                # vague, needs table
        "What is the top sector?",                        # ambiguous without table
        "List all employees with high salary",            # needs clarification
        "And what about the lowest sector?"               # relies on past question
    ]

    for query in test_queries:
        is_ans, needs_decomp, reframed, duration = agent.is_query_answerable(query, schema)
        print("\n---")
        print(f"Original Query: {query}")
        print(f"Is Answerable: {is_ans}")
        print(f"Needs Decomposition: {needs_decomp}")
        print(f"Reframed Query: {reframed}")
        print(f"Time Taken: {duration:.2f}s")
