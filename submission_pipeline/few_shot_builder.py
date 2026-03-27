"""
submission_pipeline/few_shot_builder.py
---------------------------------------
Dynamic Few-Shot Learning Module.
This system reads input rows and maps them to the expected output formats (Golden Data)
to provide the LLM with concrete examples of how to map custom fields.
"""

import os
import json
import pandas as pd
import threading
from extracting import extract_excel_data, format_df_to_llm_text

# Configuration for input-output files and their exact column name mappings
FILE_PAIRS = [
    {
        "input": "1. IVC DOE R2 (Input).xlsx",
        "output": "1. IVC DOE (Final).xlsx",
        "mapping": {
            # Maps standard expected column names to the actual columns found in Document 1
            "Risk ID": "Risk ID", "Risk Description": "Risk Description",
            "Project Stage": "Project Stage", "Project Category": "Project Category",
            "Risk Owner": "Risk Owner", "Mitigating Action": "Mitigating Action",
            "Likelihood (1-10)": "Likelihood (1-10) (pre-mitigation)",
            "Impact (1-10)": "Impact (1-10) (pre-mitigation)"
        }
    },
    {
        "input": "2. City of York Council (Input).xlsx",
        "output": "2. City of York Council (Final).xlsx",
        "mapping": {
            # Maps standard expected column names to the actual columns found in Document 2
            "Risk ID": "Risk ID", "Risk Description": "Risk Description",
            "Project Stage": "Project Stage", "Project Category": "Risk Category", # Column name differs here
            "Risk Owner": "Risk Owner", "Mitigating Action": "Mitigation",
            "Likelihood (1-10)": "Likelihood (1-10)", "Impact (1-10)": "Impact (1-10)"
        }
    }
]

# Cache variable to store the few-shot examples so they are only computed once
_CACHE_EXAMPLES_BY_COL = None
# Thread lock to prevent race conditions during concurrent cache initialization
_CACHE_LOCK = threading.Lock()

def _load_all_data():
    global _CACHE_EXAMPLES_BY_COL
    # Acquire lock before checking and initializing cache to ensure thread safety
    with _CACHE_LOCK:
        # If cache is already populated, return it immediately
        if _CACHE_EXAMPLES_BY_COL is not None:
            return _CACHE_EXAMPLES_BY_COL
            
        # Resolve absolute paths to the dataset directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "inputs"))
        output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "outputs"))
        
        # Initialize a dictionary holding empty arrays for each target column
        all_columns = FILE_PAIRS[0]["mapping"].keys()
        _CACHE_EXAMPLES_BY_COL = {col: [] for col in all_columns}
        
        print("🧠 [Few-Shot Builder] Learning mapping styles from Golden Data pairs...")
        # Iterate over predefined pairs of Input and Expected Output files
        for pair in FILE_PAIRS:
            in_path = os.path.join(input_dir, pair['input'])
            out_path = os.path.join(output_dir, pair['output'])
            
            # Skip if either the input or output file is missing in the data folders
            if not os.path.exists(in_path) or not os.path.exists(out_path): continue
                
            # Extract structured input texts via the extraction module
            df_in = extract_excel_data(in_path)
            if df_in is None: continue
            input_texts = format_df_to_llm_text(df_in)
            
            # Attempt to load the final golden output answers
            try: df_out = pd.read_excel(out_path)
            except Exception: continue
                
            # Match line-by-line length limits
            min_len = min(len(input_texts), len(df_out))
            
            # Take at most 3 examples per file to prevent LLM context-window overflow
            for i in range(min(min_len, 3)):
                in_text = input_texts[i]
                # Truncate overly long text rows to save token bandwidth
                if len(in_text) > 400: in_text = in_text[:400] + "... [TRUNCATED]"
                
                # Iterate through the column mapping constraints
                for standard_col, actual_col in pair["mapping"].items():
                    if actual_col not in df_out.columns: continue
                        
                    # Retrieve the golden expected value for the specified column
                    out_val = df_out.iloc[i].get(actual_col, "")
                    # Handle missing values rigorously
                    if pd.isna(out_val): out_val = ""
                    else:
                        out_val = str(out_val).strip()
                        # Strip .0 trailing decimals for scores (e.g., '5.0' becomes '5')
                        if standard_col in ["Likelihood (1-10)", "Impact (1-10)"] and out_val.endswith('.0'):
                            out_val = out_val[:-2]
                    
                    # Add valid ground-truth answers to the column's example cache
                    if out_val and out_val.lower() not in ['nan', 'none']:
                        _CACHE_EXAMPLES_BY_COL[standard_col].append({
                            "input_text": in_text,
                            "expected_output": out_val
                        })
                        
        return _CACHE_EXAMPLES_BY_COL

def get_few_shots_for_column(col_name):
    """Returns a formatted JSON string of few-shot examples suitable for LLM injection."""
    # Ensure data is loaded into the cache
    examples_by_col = _load_all_data()
    
    # If no examples exist for the column, gracefully return empty JSON array
    if col_name not in examples_by_col or not examples_by_col[col_name]: return "[]"
    
    # Slice to a maximum of 3 examples per column (optimal token-efficiency for LLMs)
    return json.dumps(examples_by_col[col_name][:3], ensure_ascii=False, indent=2)