import os
import time
import openai
import json
from collections import deque

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
        
        context_items = list(self.memory)[-3:]  # Last 3 for context
        
        context = "Recent Past Interactions:\n"
        for i, (query, reframed) in enumerate(reversed(context_items), 1):
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
                        You are an intelligent query reframing assistant designed to help users formulate precise questions based on a given database schema and recent conversation history. Your primary goal is to reframe user queries into clear, unambiguous natural language questions that can then be directly translated into SQL.

                        **IMPORTANT GUIDELINES:**
                        1.  **Focus on Reframing for SQL Generation:** Your output `reframed_query` should be a natural language question that directly specifies the data needed from the database, using exact table and column names from the schema when appropriate.
                        2.  **Strictly Avoid Analysis/Logic/Summarization:** Do NOT provide answers to the questions, perform logical reasoning, or generate data analysis summaries. Your task is only to reframe the *question*.
                        3.  **Leverage Context Wisely (Last 4 Interactions):** Use the `Recent Past Interactions` to infer missing details or clarify vague references in the current user question.
                        4.  **Identify Schema Elements:** Actively look for and incorporate relevant table names and column names from the provided schema into the `reframed_query`.
                        5.  **Decomposition for Complexity:** If a question is too broad or requires multiple SQL queries, set `need_decompose: "Yes"`.
                        6.  **Answerability based on Schema:** Determine `is_answerable` solely based on whether the schema contains enough information.

                        {memory_context}

                        Here is the database schema:
                        ```sql
                        {schema_string}
                        User's new question:
                        "{query}"

                        Return ONLY the JSON object in this format:

                        {{
                        "is_answerable": "Yes" or "No",
                        "need_decompose": "Yes" or "No",
                        "reframed_query": "..."
                        }}
                        """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )

            answer = response.choices[0].message.content.strip()
            result = json.loads(answer)

            is_answerable_str = result.get("is_answerable", "").strip().lower()
            need_decompose_str = result.get("need_decompose", "").strip().lower()
            reframed_query = result.get("reframed_query", query)

            is_answerable = is_answerable_str == "yes"
            need_decompose = need_decompose_str == "yes"

            self.add_to_memory(query, reframed_query)

            return is_answerable, need_decompose, reframed_query, time.time() - start_time
    
        except json.JSONDecodeError:
            print(f"[ERROR] JSON parsing failed. Raw content: {answer}")
            return False, False, f"Parsing failed for query: '{query}'", time.time() - start_time
        except openai.APIError as e:
            print(f"[ERROR] OpenAI API Error: {e}")
            return False, False, f"API Error: {str(e)}", time.time() - start_time
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            return False, False, f"Error: {str(e)}", time.time() - start_time

if __name__ == "__main__":
    SCHEMA = """
    Table: nationals_account_isic (
    sector text,
    2015 bigint,
    2016 bigint,
    2017 bigint,
    2018 bigint,
    2019 bigint,
    2020 bigint,
    2021 bigint,
    2022 bigint,
    2023 bigint
    );
    Table: orders (
    order_id integer,
    customer_id integer,
    order_date text
    );
    """

    
    selector = SelectorAgent(max_memory=4)

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        is_ans, needs_decomp, reframed, duration = selector.is_query_answerable(user_query, SCHEMA)

        print("\n--- Analysis Result ---")
        print(f"Original Query: {user_query}")
        print(f"Reframed Query: {reframed}")
        print(f"Is Answerable: {is_ans}")
        print(f"Needs Decomposition: {needs_decomp}")
        print(f"Time Taken: {duration:.2f} seconds")
        print(f"Memory (last {len(selector.memory)}): {list(selector.memory)}")
   






