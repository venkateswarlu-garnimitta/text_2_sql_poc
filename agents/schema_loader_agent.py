import time
from database import get_database_schema_string

class SchemaLoaderAgent:
    """
    Agent responsible for loading the database schema.
    """
    @staticmethod
    def load_schema_from_db(db_path: str = None) -> tuple[str, float]:
        """
        Loads the database schema string from the specified SQLite database.

        Args:
            db_path (str): The path to the SQLite database file.

        Returns:
            tuple[str, float]: A tuple containing the schema string and the time taken in seconds.
        """
        start_time = time.time()
        schema_string = get_database_schema_string()
        end_time = time.time()
        time_taken = end_time - start_time
        return schema_string, time_taken

# Example usage (for testing)
if __name__ == '__main__':
    # Ensure you have a 'data' directory and 'my_database.db' inside it
    # with at least one table for this to work.
    schema, duration = SchemaLoaderAgent.load_schema_from_db("data/my_database.db")
    print("Schema Loaded:")
    print(schema)
    print(f"Time taken: {duration:.4f} seconds")
