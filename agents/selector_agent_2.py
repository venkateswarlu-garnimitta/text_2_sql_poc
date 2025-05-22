import os
import time
import openai
import json
from collections import deque
import re

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
        
        # Get only the last 3 interactions for context
        context_items = list(self.memory)[-3:] 
        
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
3.  **Leverage Context Wisely (Last 4 Interactions):** Use the `Recent Past Interactions` to infer missing details or clarify vague references in the current user question. For example, if a previous query asked for "top 10 rows in table1" and the current query is "same in table2", reframe it as "top 10 rows in table2".
4.  **Identify Schema Elements:** Actively look for and incorporate relevant table names and column names from the provided schema into the `reframed_query` to make it explicit.
5.  **Decomposition for Complexity:** If a question is too broad or requires multiple, distinct SQL queries (e.g., "Show me sales trends and customer demographics"), indicate `need_decompose: "Yes"`.
6.  **Answerability based on Schema:** Determine `is_answerable` solely based on whether the information *could theoretically be retrieved* from the provided schema. If a query asks for data not present in the schema, it is "No".

{memory_context}

Here is the database schema:
```sql
{schema_string}
User's new question:
"{query}"

Based on the schema and recent interactions, please perform the following tasks and return the result in JSON format:

Tasks:

is_answerable: "Yes" if the question can be answered from the provided schema, "No" otherwise.
need_decompose: "Yes" if the question is complex and clearly needs to be broken down into multiple sub-questions/SQL queries, "No" otherwise.
reframed_query: The most precise natural language question, incorporating schema elements and context, that can be directly used to generate a SQL query. If the original query is already perfect, return it as is.
Return ONLY the JSON object. Do not include any conversational text, explanations, or markdown outside the JSON.

JSON Format Example:

JSON

{{
  "is_answerable": "Yes",
  "need_decompose": "No",
  "reframed_query": "What are the top 10 rows from the 'employees' table?"
}}
"""

# ```

# ## Critical Fixes Applied:

# 🚫 **SQL Reality Check**: Added explicit logic to identify questions that require explanation, analysis, or external knowledge

# 📋 **Clear Categories**: Defined exactly what IS and ISN'T answerable by SQL with examples

# 🔍 **Smart Filtering**: Will now correctly identify "explain why there is a drop" as NOT answerable since SQL can't provide explanations

# 🎯 **Realistic Assessment**: Focuses on what SQL databases can actually do vs. what they cannot

# ⚡ **Prevents Failures**: Will stop passing impossible queries to your SQL agent

# Now your examples will be handled correctly:
# - ❌ "explain why there is a drop in orders" → `"is_answerable": "No"` 
# - ❌ "who is the prime minister of india" → `"is_answerable": "No"`
# - ✅ "show me order data" → `"is_answerable": "Yes"`

# This prevents your SQL agent from receiving impossible queries and failing!

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )

            answer = response.choices[0].message.content.strip()
            # st.write(f"Raw response: {answer}") # Debug print for Streamlit

            try:
                result = json.loads(answer)

                is_answerable_str = result.get("is_answerable", "").strip().lower()
                need_decompose_str = result.get("need_decompose", "").strip().lower()
                reframed_query = result.get("reframed_query", query)
                intent_analysis = result.get("intent_analysis","").strip().lower()
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(intent_analysis)
                is_answerable = is_answerable_str == "yes"
                need_decompose = need_decompose_str == "yes"

                # Store both original and reframed for clarity in memory, but context uses only reframed
                self.add_to_memory(query, reframed_query)

                return is_answerable, need_decompose, reframed_query, time.time() - start_time

            except json.JSONDecodeError:
                st.error(f"JSON parsing failed. Raw content: {answer}")
                return False, False, f"Parsing failed for query: '{query}'", time.time() - start_time
            except AttributeError:
                st.error(f"AttributeError: Likely missing keys in JSON response. Raw content: {answer}")
                return False, False, f"Missing keys in response for query: '{query}'", time.time() - start_time

        except openai.APIError as e:
            st.error(f"OpenAI API Error: {e}")
            return False, False, f"API Error: {str(e)}", time.time() - start_time
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return False, False, f"Error: {str(e)}", time.time() - start_time

# --- Streamlit Application ---

# st.set_page_config(page_title="Interactive Selector Agent")
# st.title("📊 Interactive SQL Query Selector Agent")

# # Define your schema
# SCHEMA = """
# Table: nationals_account_isic (
#   sector text,
#   2015 bigint,
#   2016 bigint,
#   2017 bigint,
#   2018 bigint,
#   2019 bigint,
#   2020 bigint,
#   2021 bigint,
#   2022 bigint,
#   2023 bigint
# );
# Table: orders (
#   order_id integer,
#   customer_id integer,
#   order_date text
# );
# """

# # Initialize SelectorAgent in session_state if it doesn't exist
# # This ensures the same agent instance is used across reruns
# if 'selector_agent' not in st.session_state:
#     st.session_state.selector_agent = SelectorAgent(max_memory=4)
# if 'query_count' not in st.session_state:
#     st.session_state.query_count = 0
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Display database schema
# with st.expander("View Database Schema"):
#     st.code(SCHEMA, language='sql')

# # Display chat history
# st.subheader("Conversation History")
# for i, entry in enumerate(st.session_state.chat_history):
#     st.markdown(f"**User Query {i+1}:** {entry['original_query']}")
#     st.markdown(f"**Reframed Query:** {entry['reframed_query']}")
#     st.markdown(f"Is Answerable: {entry['is_answerable']}, Needs Decomposition: {entry['needs_decomposition']}")
#     st.markdown("---") # Divider

# # User input
# user_query = st.text_input("Enter your query:", key="user_query_input")

# if user_query: # Process query only if input is not empty
#     st.session_state.query_count += 1
#     current_query_num = st.session_state.query_count

#     # Get the agent instance from session_state
#     agent = st.session_state.selector_agent

#     st.info(f"**Processing Query {current_query_num}:** \"{user_query}\"")
#     st.text(f"Current Memory Context:\n{agent.get_memory_context()}")

#     is_ans, needs_decomp, reframed, duration = agent.is_query_answerable(user_query, SCHEMA)

#     st.subheader(f"Analysis for Query {current_query_num}")
#     st.write(f"**Original Query:** {user_query}")
#     st.write(f"**Is Answerable:** {is_ans}")
#     st.write(f"**Needs Decomposition:** {needs_decomp}")
#     st.write(f"**Reframed Query:** {reframed}")
#     st.write(f"**Time Taken:** {duration:.2f}s")
#     st.write(f"**Memory content (original, reframed pairs):** {list(agent.memory)}")

#     # Add to chat history for display
#     st.session_state.chat_history.append({
#         "original_query": user_query,
#         "reframed_query": reframed,
#         "is_answerable": is_ans,
#         "needs_decomposition": needs_decomp
#     })

#     # Clear the input box after submission
#     # This might require a trick to clear the text_input widget
#     # For simplicity here, the text remains until cleared by user or new input
#     # A common pattern is to use a button to trigger action and clear input programmatically
#     # e.g., if st.button("Submit Query"): ... then clear input_key
#     # For text_input directly, clearing it is tricky as it's stateful.
#     # We'll rely on the user typing a new query.

# # Optional: Button to clear chat history and reset agent
# if st.button("Clear Chat History and Reset Agent"):
#     st.session_state.selector_agent = SelectorAgent(max_memory=4)
#     st.session_state.query_count = 0
#     st.session_state.chat_history = []
#     st.rerun() # Rerun the app to reflect the changes