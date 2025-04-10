import sys
import json
import pandas as pd
from mcp.server.fastmcp import FastMCP
# Import the preprocessing function from your existing script
from preprocess import preprocess_dataframe

# Define the MCP server
mcp = FastMCP('preprocess')

@mcp.tool()
def run_preprocessing(dataframe_json: str):
    """Processes a DataFrame passed as a JSON string (orient='split').

    Args:
        dataframe_json: A JSON string representing the pandas DataFrame (orient='split').

    Returns:
        A JSON string representing the processed pandas DataFrame (orient='split').
    """
    print("--- MCP Tool: run_preprocessing started ---", file=sys.stderr)
    try:
        # Deserialize JSON string to DataFrame
        print("--- MCP Tool: Deserializing input JSON ---", file=sys.stderr)
        input_df = pd.read_json(dataframe_json, orient='split')
        print(f"--- MCP Tool: Input DataFrame shape: {input_df.shape} ---", file=sys.stderr)

        # Perform preprocessing
        print("--- MCP Tool: Calling preprocess_dataframe ---", file=sys.stderr)
        processed_df = preprocess_dataframe(input_df)
        print(f"--- MCP Tool: Processed DataFrame shape: {processed_df.shape} ---", file=sys.stderr)

        # Serialize the processed DataFrame back to JSON
        print("--- MCP Tool: Serializing output JSON ---", file=sys.stderr)
        # Use orient='split' and index=False for consistency, adjust if needed
        output_json = processed_df.to_json(orient='split', index=False)

        print("--- MCP Tool: Preprocessing successful ---", file=sys.stderr)
        return output_json

    except json.JSONDecodeError as e:
        print(f"--- MCP Tool Error: Invalid JSON input: {e} ---", file=sys.stderr)
        # Consider how to best report errors back via MCP context if needed
        # For now, re-raising might be okay or return an error structure
        raise ValueError(f"Invalid JSON input: {e}") from e
    except Exception as e:
        print(f"--- MCP Tool Error: Preprocessing failed: {e} ---", file=sys.stderr)
        # Re-raise to indicate failure to the MCP client
        raise RuntimeError(f"Preprocessing failed: {e}") from e

if __name__ == "__main__":
    print("--- Starting MCP Preprocessing Server (stdio) ---", file=sys.stderr)
    mcp.run(transport='stdio')
    print("--- MCP Preprocessing Server stopped ---", file=sys.stderr) 