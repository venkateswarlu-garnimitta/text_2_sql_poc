import os
import time
import openai
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
class VisualizationAgent:
    """
    Agent responsible for suggesting the best visualization type for the query result
    and generating clean, professional insights about the data.
    """
    @staticmethod
    def suggest_visualization(query: str, df: pd.DataFrame) -> tuple[str, float, str, str]:
        start_time = time.time()
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Metadata preparation
            columns_info = {col: str(df[col].dtype) for col in df.columns}
            columns_info_str = json.dumps(columns_info, indent=2)
            sample_data_str = json.dumps(df.head(100).to_dict(orient='records'), indent=2, default=str)

            total_rows = len(df)
            total_columns = len(df.columns)
            total_nulls = int(df.isnull().sum().sum())

            valid_types = ["Table", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap"]
            valid_types_str = ", ".join(valid_types)

            prompt = f"""
You are a professional **data visualization assistant**. Your role is to analyze structured data and suggest the best visualization along with clear, well-written summaries.

Please follow the strict structure below and return a **valid JSON** only, with no extra commentary.

### Context
- **Query:** "{query}"
- **Column Types:** {columns_info_str}
- **Sample Data (first 100 rows):** {sample_data_str}

### Output Format
Respond strictly in this JSON format:

{{
  "visualization_type": "Choose the best fit from: {valid_types_str}. Keep it simple and relevant.",
  
  "insights": "Write 2–4 bullet points in Markdown. Follow these rules:
  - Start with ### Summary
  - Keep language neutral, professional, and positive.
  - No negative, sensitive, biased, or personal commentary.
  - Use correct grammar and spelling.
  - No unnecessary repetition.
  - Use AED formatting (e.g., AED 10.5k).
  - Use good markdown for better visibility like using bold and any other types whenever necessary",

  "data_insights": "Return the following technical metadata in markdown **tabular format**:\n\n\
| Metric              | Value        |\n\
|---------------------|--------------|\n\
| Rows                | {total_rows} |\n\
| Columns             | {total_columns} |\n\
| Total Null Values   | {total_nulls} |\n\
| Column Data Types   | {json.dumps(columns_info)} |\n\n\
Also include a short paragraph (2–3 lines) with factual insights like column patterns or notable types. Do not repeat summary insights here."
}}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            response_content = response.choices[0].message.content.strip()
            print("LLM Response:\n", response_content)

            # Safe parsing
            visualization_type = "Table"
            insights = "No insights available."
            data_insights = "No data insights found."

            try:
                response_json = json.loads(response_content)
                visualization_type = response_json.get("visualization_type", visualization_type).strip()
                insights = response_json.get("insights", insights).strip()
                data_insights = response_json.get("data_insights", data_insights).strip()
            except json.JSONDecodeError:
                print("Failed to parse JSON. Check LLM formatting.")
                insights = insights.encode().decode("unicode_escape").strip('`"\n {}')
                data_insights = data_insights.encode().decode("unicode_escape").strip('`"\n {}')

            best_match = "Table"
            for vtype in valid_types:
                if vtype.lower() in visualization_type.lower():
                    best_match = vtype
                    break

            time_taken = time.time() - start_time
            return best_match, time_taken, insights, data_insights

        except Exception as e:
            time_taken = time.time() - start_time
            print(f"Error in VisualizationAgent: {str(e)}")
            return "Table", time_taken, "Could not generate insights due to an error.", "No data insights found."

# Example usage
if __name__ == '__main__':
    test_df = pd.DataFrame({
        'department': ['Sales', 'IT', 'HR', 'IT'],
        'salary': [50000, 60000, 52000, 61000]
    })
    query = "What is the average salary by department?"

    if os.getenv("OPENAI_API_KEY"):
        vis_type, duration, insights, data_insights = VisualizationAgent.suggest_visualization(query, test_df)
        print("Visualization Type:", vis_type)
        print("Time Taken:", duration)
        print("\nInsights (Markdown):\n", insights)
        print("\nData Insights (Markdown Table):\n", data_insights)
    else:
        print("Missing OPENAI_API_KEY.")
