from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
import json
import time
import base64
import sqlparse
from pathlib import Path
import re
import os
from dotenv import load_dotenv

# Assuming these are separate Python files/modules based on your Streamlit app
# Make sure these modules (agents, database, file_uploads, pdf) are available
from agents.schema_loader_agent import SchemaLoaderAgent
from agents.selector_agent import SelectorAgent
from agents.decomposer_agent import DecomposerAgent
from agents.refiner_agent import RefinerAgent
from agents.visualization_agent import VisualizationAgent
from database import execute_sql_query_db
from fast_api_file_upload import insert_data, create_table_if_not_exists, clean_column_name, table_exists # Import table_exists
from pdf import PDFReportGenerator # Assuming PDFReportGenerator is a class

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NL2SQL Assistant API",
    description="API for converting natural language queries to SQL and visualizing results."
)

# --- In-memory storage for schema (for simplicity, consider a more robust solution for production) ---
db_schema_string: str = ""

@app.on_event("startup")
async def startup_event():
    """Load the database schema when the FastAPI application starts."""
    global db_schema_string
    print("Loading database schema on startup...")
    try:
        schema_string, time_taken = SchemaLoaderAgent.load_schema_from_db()
        if schema_string and "Error loading database schema" not in schema_string:
            db_schema_string = schema_string
            print("Database schema loaded successfully.")
        else:
            print(f"Failed to load database schema: {schema_string}")
            # Depending on criticality, you might raise an error here
            # raise RuntimeError("Failed to load database schema")
    except Exception as e:
        print(f"An error occurred during schema loading: {e}")
        # raise RuntimeError(f"An error occurred during schema loading: {e}")


# --- Pydantic Models for Request/Response ---

class ProcessQueryRequest(BaseModel):
    """Request body for the process query endpoint."""
    query: str

class ProcessQueryResponse(BaseModel):
    """Response body for the process query endpoint."""
    original_query: str
    sql_query: str
    result_data: list[dict] | None = None # List of dictionaries for JSON representation
    suggested_visualization: str | None = None
    data_insights: str | None = None
    processing_steps: list[dict]
    total_time_seconds: float
    error: str | None = None

class UploadCSVResponse(BaseModel):
    """Response body for the upload CSV endpoint."""
    filename: str
    table_name: str
    status: str
    message: str
    error: str | None = None

class GeneratePDFRequest(BaseModel):
    """Request body for generating PDF."""
    original_query: str
    sql_query: str
    result_data: list[dict] # Data to include in the PDF
    summary: str | None = None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Basic root endpoint."""
    return {"message": "NL2SQL Assistant API is running."}

@app.get("/schema")
async def get_schema():
    """Endpoint to retrieve the currently loaded database schema."""
    if not db_schema_string or "Error loading database schema" in db_schema_string:
        raise HTTPException(status_code=500, detail="Database schema not loaded or an error occurred.")
    return {"schema": db_schema_string}

@app.post("/upload-csv", response_model=UploadCSVResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        table_name = clean_column_name(file.filename.replace(".csv", ""))
        if not table_name:
            raise HTTPException(status_code=400, detail="Invalid table name from filename.")

        create_table_if_not_exists(table_name, df)
        insert_data(df, table_name)

        return UploadCSVResponse(
            filename=file.filename,
            table_name=table_name,
            status="success",
            message=f"File '{file.filename}' uploaded and data inserted."
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    
    
@app.post("/process-query", response_model=ProcessQueryResponse)
async def process_natural_language_query(request: ProcessQueryRequest):
    """
    Endpoint to process a natural language query, generate SQL, execute it,
    and suggest a visualization.
    """
    start_time = time.time()
    query = request.query
    processing_steps = []
    result_df = None
    sql_query = ""
    suggested_visualization = None
    data_insights = None
    error_message = None

    if not db_schema_string or "Error loading database schema" in db_schema_string:
        error_message = "Database schema not loaded or an error occurred. Cannot process query."
        processing_steps.append({
            "agent": "Schema Loader",
            "status": "❌ Schema not loaded",
            "details": error_message,
            "time_taken": 0.0
        })
        end_time = time.time()
        return ProcessQueryResponse(
            original_query=query,
            sql_query="",
            result_data=None,
            suggested_visualization=None,
            data_insights=None,
            processing_steps=processing_steps,
            total_time_seconds=(end_time - start_time),
            error=error_message
        )

    # Step 1: Schema Loading (Check if already loaded)
    processing_steps.append({
        "agent": "Schema Loader",
        "status": "✅ Schema already loaded from database",
        "details": "Database schema is available.",
        "time_taken": 0.0 # Time recorded during initial load
    })

    # Step 2: Selector Agent - Check if query is answerable
    start_time_selector = time.time()
    try:
        is_answerable, need_decompose, _ = SelectorAgent.is_query_answerable(
            query, db_schema_string
        )
        time_taken_selector = time.time() - start_time_selector
        status_selector = "✅ Query is answerable" if is_answerable else "❌ Query is not answerable"
        processing_steps.append({
            "agent": "Selector Agent",
            "status": status_selector,
            "details": f"Need decomposer: {need_decompose}",
            "time_taken": time_taken_selector
        })

        if not is_answerable:
            error_message = "Query cannot be answered with the current database schema."
            end_time = time.time()
            return ProcessQueryResponse(
                original_query=query,
                sql_query="",
                result_data=None,
                suggested_visualization=None,
                data_insights=None,
                processing_steps=processing_steps,
                total_time_seconds=(end_time - start_time),
                error=error_message
            )
    except Exception as e:
        time_taken_selector = time.time() - start_time_selector
        error_message = f"Error during Selector Agent processing: {e}"
        processing_steps.append({
            "agent": "Selector Agent",
            "status": "❌ Error",
            "details": error_message,
            "time_taken": time_taken_selector
        })
        end_time = time.time()
        return ProcessQueryResponse(
            original_query=query,
            sql_query="",
            result_data=None,
            suggested_visualization=None,
            data_insights=None,
            processing_steps=processing_steps,
            total_time_seconds=(end_time - start_time),
            error=error_message
        )


    # Step 3: Decomposer Agent - Break down the query
    decomposition = None
    if need_decompose:
        start_time_decomposer = time.time()
        try:
            decomposition, _ = DecomposerAgent.decompose_query(
                query, db_schema_string
            )
            time_taken_decomposer = time.time() - start_time_decomposer
            processing_steps.append({
                "agent": "Decomposer Agent",
                "status": "✅ Query decomposed",
                "details": decomposition,
                "time_taken": time_taken_decomposer
            })
        except Exception as e:
             time_taken_decomposer = time.time() - start_time_decomposer
             error_message = f"Error during Decomposer Agent processing: {e}"
             processing_steps.append({
                 "agent": "Decomposer Agent",
                 "status": "❌ Error",
                 "details": error_message,
                 "time_taken": time_taken_decomposer
             })
             end_time = time.time()
             return ProcessQueryResponse(
                 original_query=query,
                 sql_query="",
                 result_data=None,
                 suggested_visualization=None,
                 data_insights=None,
                 processing_steps=processing_steps,
                 total_time_seconds=(end_time - start_time),
                 error=error_message
             )


    # Step 4: Refiner Agent - Generate SQL
    start_time_refiner = time.time()
    try:
        sql_query, _ = RefinerAgent.generate_sql(
            query, db_schema_string, decomposition
        )
        time_taken_refiner = time.time() - start_time_refiner
        processing_steps.append({
            "agent": "Refiner Agent",
            "status": "✅ SQL generated",
            "details": sql_query,
            "time_taken": time_taken_refiner
        })
    except Exception as e:
         time_taken_refiner = time.time() - start_time_refiner
         error_message = f"Error during Refiner Agent processing: {e}"
         processing_steps.append({
             "agent": "Refiner Agent",
             "status": "❌ Error",
             "details": error_message,
             "time_taken": time_taken_refiner
         })
         end_time = time.time()
         return ProcessQueryResponse(
             original_query=query,
             sql_query=sql_query, # Include generated SQL even if execution fails
             result_data=None,
             suggested_visualization=None,
             data_insights=None,
             processing_steps=processing_steps,
             total_time_seconds=(end_time - start_time),
             error=error_message
         )


    # Execute the query against the database
    start_time_exec = time.time()
    try:
        result_df = execute_sql_query_db(sql_query)
        time_taken_exec = time.time() - start_time_exec

        if result_df is not None:
            processing_steps.append({
                "agent": "Database Execution",
                "status": "✅ Query executed successfully",
                "details": f"Query returned {len(result_df)} rows and {len(result_df.columns)} columns.",
                "time_taken": time_taken_exec
            })
        else:
            error_message = "SQL query execution failed or returned None."
            processing_steps.append({
                "agent": "Database Execution",
                "status": "❌ Query execution failed",
                "details": error_message,
                "time_taken": time_taken_exec
            })
            end_time = time.time()
            return ProcessQueryResponse(
                original_query=query,
                sql_query=sql_query,
                result_data=None,
                suggested_visualization=None,
                data_insights=None,
                processing_steps=processing_steps,
                total_time_seconds=(end_time - start_time),
                error=error_message
            )
    except Exception as e:
        time_taken_exec = time.time() - start_time_exec
        error_message = f"Error during SQL query execution: {e}"
        processing_steps.append({
            "agent": "Database Execution",
            "status": "❌ Error",
            "details": error_message,
            "time_taken": time_taken_exec
        })
        end_time = time.time()
        return ProcessQueryResponse(
            original_query=query,
            sql_query=sql_query,
            result_data=None,
            suggested_visualization=None,
            data_insights=None,
            processing_steps=processing_steps,
            total_time_seconds=(end_time - start_time),
            error=error_message
        )


    # Step 5: Visualization Agent (only if query execution was successful and returned data)
    if result_df is not None and not result_df.empty:
        start_time_vis = time.time()
        try:
            vis_type, _, insights = VisualizationAgent.suggest_visualization(query, result_df)
            suggested_visualization = vis_type

            # Attempt to clean and format insights
            insights = insights.strip().strip('```').strip().strip('}').strip().strip(']').strip().strip(': [')
            try:
                # Decode potential unicode escape sequences
                formatted_insights = insights.encode().decode("unicode_escape")
            except Exception:
                formatted_insights = insights # Fallback if decoding fails

            data_insights = formatted_insights

            time_taken_vis = time.time() - start_time_vis
            processing_steps.append({
                "agent": "Visualization Agent",
                "status": "✅ Visualization suggested with insights",
                "details": f"Suggested visualization: `{vis_type}`\n\nInsights:\n\n{data_insights}",
                "time_taken": time_taken_vis
            })
        except Exception as e:
            time_taken_vis = time.time() - start_time_vis
            error_message = f"Error during Visualization Agent processing: {e}"
            processing_steps.append({
                "agent": "Visualization Agent",
                "status": "❌ Error",
                "details": error_message,
                "time_taken": time_taken_vis
            })
            # Continue without visualization/insights if this step fails
            suggested_visualization = "Table" # Default to table
            data_insights = "Could not generate insights."

    elif result_df is not None and result_df.empty:
        processing_steps.append({
            "agent": "Visualization Agent",
            "status": "ℹ️ No data for visualization",
            "details": "Query returned an empty result set.",
            "time_taken": 0.0
        })
        suggested_visualization = "Table" # Default to table even with no data
        data_insights = "Query returned no data."

    end_time = time.time()
    total_time = end_time - start_time

    # Convert DataFrame to a list of dictionaries for JSON response
    result_data_list = result_df.to_dict(orient='records') if result_df is not None else None

    return ProcessQueryResponse(
        original_query=query,
        sql_query=sql_query,
        result_data=result_data_list,
        suggested_visualization=suggested_visualization,
        data_insights=data_insights,
        processing_steps=processing_steps,
        total_time_seconds=total_time,
        error=error_message # This will be None if no error occurred
    )

@app.post("/generate-pdf")
async def generate_pdf_report(request: GeneratePDFRequest):
    """
    Endpoint to generate a PDF report from query results.
    Expects result_data as a list of dictionaries.
    """
    if not request.result_data:
        raise HTTPException(status_code=400, detail="No result data provided for PDF generation.")

    try:
        # Convert list of dicts back to DataFrame
        result_df = pd.DataFrame(request.result_data)

        # Generate PDF report
        pdf_bytes = PDFReportGenerator.generate_report(
            query=request.original_query,
            sql_query=request.sql_query,
            result_df=result_df,
            summary=request.summary,
            output_path=None  # Generate in-memory bytes
        )

        # Return PDF as a StreamingResponse
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=query_report.pdf"}
        )

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the PDF: {e}")

@app.post("/export-csv")
async def export_csv(result_data: list[dict]):
    """
    Endpoint to export data as CSV.
    Expects result_data as a list of dictionaries.
    """
    if not result_data:
        raise HTTPException(status_code=400, detail="No data provided for CSV export.")

    try:
        df = pd.DataFrame(result_data)
        csv_content = df.to_csv(index=False).encode('utf-8')

        return StreamingResponse(
            io.BytesIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=query_result.csv"}
        )
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while exporting to CSV: {e}")

@app.post("/export-excel")
async def export_excel(result_data: list[dict]):
    """
    Endpoint to export data as Excel.
    Expects result_data as a list of dictionaries.
    """
    if not result_data:
        raise HTTPException(status_code=400, detail="No data provided for Excel export.")

    try:
        df = pd.DataFrame(result_data)
        excel_content = io.BytesIO()
        df.to_excel(excel_content, index=False, engine='openpyxl')
        excel_content.seek(0) # Move to the beginning of the BytesIO object

        return StreamingResponse(
            excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=query_result.xlsx"}
        )
    except Exception as e:
        print(f"Error exporting Excel: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while exporting to Excel: {e}")