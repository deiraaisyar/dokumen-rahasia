import math # Used for mathematical operations like calculating square roots for cosine similarity
import re # Used for regular expressions to clean and normalize text
from collections import Counter # Used to efficiently count word frequencies (tokens)
from pathlib import Path # Used for cross-platform file path handling

import pandas as pd # Used for data manipulation and reading tabular files


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel based on file extension."""
    suffix = path.suffix.lower() # Extract the file extension in lowercase
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path) # Use pandas Excel reader for .xlsx/.xls
    return pd.read_csv(path) # Default back to CSV reader for other files


def _normalize_text(value) -> str:
    """Normalize text for stable comparison across formatting differences."""
    if pd.isna(value): # Check if value is null/NaN
        return "" # Return empty string instead of "nan"
    text = str(value).strip().lower() # Convert safely to string, remove leading/trailing spaces, and make lowercase
    text = re.sub(r"\s+", " ", text) # Replace all multiple spaces, tabs, or newlines with a single space
    return text


def _token_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity on token frequency vectors without external deps."""
    # Cosine similarity checks how similar two sentences are based on the words they share
    a = _normalize_text(text_a) # Clean first text
    b = _normalize_text(text_b) # Clean second text
    if not a and not b:
        return 1.0 # If both texts are completely empty, they are a perfect match
    if not a or not b:
        return 0.0 # If only one of them is empty, they don't match at all

    ca = Counter(a.split()) # Count frequency of each word in text A
    cb = Counter(b.split()) # Count frequency of each word in text B
    vocab = set(ca) | set(cb) # Create a combined vocabulary list from both texts
    
    # Calculate the dot product (sum of word matches across both texts)
    dot = sum(ca[t] * cb[t] for t in vocab)
    
    # Calculate the magnitude (vector length) of each text
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Prevent division by zero if somehow vectors are empty
        
    return dot / (norm_a * norm_b) # Apply Cosine Similarity formula: (A . B) / (||A|| * ||B||)


def _to_float(value):
    """Convert values to float safely for numeric evaluation."""
    try:
        if pd.isna(value):
            return None # Ignore nulls in numerical calculation
        text = str(value).strip()
        if not text:
            return None # Ignore blank strings
        return float(text) # Try to cast string object strictly to float
    except Exception:
        return None # If it fails (e.g., cell contains text "High"), return None safely


def _align_rows(gen_df: pd.DataFrame, ref_df: pd.DataFrame, key_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align generated and reference rows by key column; fallback to row order when needed."""
    # Checks if the designated primary key exists in both the generated output and ground-truth reference
    if key_column in gen_df.columns and key_column in ref_df.columns:
        gen = gen_df.copy()
        ref = ref_df.copy()
        
        # Create temporary columns '__k' to ensure the keys align correctly even if there are whitespace issues
        gen["__k"] = gen[key_column].astype(str).str.strip()
        ref["__k"] = ref[key_column].astype(str).str.strip()
        
        # Merge both datasets strictly where the keys match (inner join)
        merged = gen.merge(ref, on="__k", how="inner", suffixes=("_gen", "_ref"))
        
        if len(merged) > 0:
            # Separate the merged dataframe back into purely generated columns and purely reference columns
            gen_cols = [c for c in merged.columns if c.endswith("_gen")]
            ref_cols = [c for c in merged.columns if c.endswith("_ref")]
            gen_aligned = merged[gen_cols].copy()
            ref_aligned = merged[ref_cols].copy()
            
            # Remove the "_gen" and "_ref" suffixes to restore original header names
            gen_aligned.columns = [c[:-4] for c in gen_aligned.columns]
            ref_aligned.columns = [c[:-4] for c in ref_aligned.columns]
            return gen_aligned.reset_index(drop=True), ref_aligned.reset_index(drop=True)

    # Fallback Mechanism: If key column is missing, simply truncate both files to match the shortest one
    n = min(len(gen_df), len(ref_df))
    return gen_df.iloc[:n].reset_index(drop=True), ref_df.iloc[:n].reset_index(drop=True)


def _evaluate_pair(doc_name: str, gen_path: Path, ref_path: Path, key_column: str) -> dict:
    """Evaluate one generated doc against its reference preprocessed output."""
    gen_df = _read_table(gen_path) # Load AI Generated data
    ref_df = _read_table(ref_path) # Load Ground-Truth Target data
    gen_df, ref_df = _align_rows(gen_df, ref_df, key_column) # Ensure rows are matched 1:1

    # Keep only column headers that exist in both files
    common_cols = [c for c in gen_df.columns if c in ref_df.columns]
    
    if not common_cols:
        # Failsafe if files share absolutely no common headers
        return {
            "doc": doc_name,
            "rows_compared": 0,
            "text_cosine_avg": 0.0,
            "exact_match_rate": 0.0,
            "numeric_mae": None,
            "numeric_within_0_1_rate": None,
            "overall_score": 0.0,
        }

    text_scores = []
    exact_hits = 0
    exact_total = 0
    num_abs_errors = []
    num_within = 0
    num_total = 0

    for col in common_cols:
        gen_series = gen_df[col] # Get the current column for Generated data
        ref_series = ref_df[col] # Get the current column for Reference data

        # Attempt to cast entire columns to numeric
        gen_num = gen_series.apply(_to_float)
        ref_num = ref_series.apply(_to_float)
        
        # A mask to check which pairs of rows contain valid numbers on both sides
        numeric_mask = gen_num.notna() & ref_num.notna()

        # If there are any numbers, we evaluate them as Mathematical Errors (Mean Absolute Error - MAE)
        if numeric_mask.any():
            for gv, rv in zip(gen_num[numeric_mask], ref_num[numeric_mask]):
                err = abs(gv - rv) # Calculate absolute diff (e.g. Generated says 8, Ref says 5. Error is 3)
                num_abs_errors.append(err)
                if err <= 0.1: # Threshold allowance for small rounding differences
                    num_within += 1
                num_total += 1
        
        # If the column has no numbers, we evaluate it as Natural Text using Cosine Similarity
        else:
            for gv, rv in zip(gen_series, ref_series):
                gtxt = _normalize_text(gv)
                rtxt = _normalize_text(rv)
                text_scores.append(_token_cosine_similarity(gtxt, rtxt)) # Save similarity float 0.0 - 1.0
                exact_hits += int(gtxt == rtxt) # If sentences match exactly (100%), count as an exact hit
                exact_total += 1

    # Aggregate Evaluation Results across all rows and columns
    text_cosine_avg = sum(text_scores) / len(text_scores) if text_scores else 1.0 
    exact_match_rate = exact_hits / exact_total if exact_total else 1.0
    numeric_mae = sum(num_abs_errors) / len(num_abs_errors) if num_abs_errors else None
    numeric_within_rate = num_within / num_total if num_total else None

    # Calculate overall numerical deduction score. A Max error scaling of 10.0 (drops score to 0 if MAE > 10)
    if numeric_mae is None:
        numeric_score = 1.0
    else:
        numeric_score = max(0.0, 1.0 - (numeric_mae / 10.0))

    # Weighting logic for Final Overall Pipeline Score (50% Text Similarities, 25% Text Exact Match, 25% Numeric Match)
    overall_score = 0.5 * text_cosine_avg + 0.25 * exact_match_rate + 0.25 * numeric_score

    # Return a structured dictionary for this document's evaluation
    return {
        "doc": doc_name,
        "rows_compared": len(gen_df),
        "text_cosine_avg": round(text_cosine_avg, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "numeric_mae": None if numeric_mae is None else round(numeric_mae, 4),
        "numeric_within_0_1_rate": None if numeric_within_rate is None else round(numeric_within_rate, 4),
        "overall_score": round(overall_score, 4),
    }


def run_evaluating():
    """Run evaluation for generated doc1-3 against preprocessed reference outputs."""
    # Define mapping paths: the AI answers vs the preprocessed target answers from Judges
    mappings = [
        {
            "doc": "doc1",
            "gen": Path("./generated_outputs/1. IVC DOE (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/1. IVC DOE (Final).csv"),
            "key": "Risk ID",
        },
        {
            "doc": "doc2",
            "gen": Path("./generated_outputs/2. City of York Council (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/2. City of York Council (Final).csv"),
            "key": "Risk ID",
        },
        {
            "doc": "doc3",
            "gen": Path("./generated_outputs/3. Digital Security IT Sample Register (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/3. Digital Security IT Sample Register (Final).csv"),
            "key": "Number", # Notice: Doc 3 uses "Number" instead of "Risk ID" for row alignment mapping
        },
    ]

    results = []
    # Loop over mapped documents and evaluate pairs
    for item in mappings:
        if not item["gen"].exists() or not item["ref"].exists():
            print(f"Skip {item['doc']}: missing file")
            continue
        
        # Dispatch the evaluation logic function
        result = _evaluate_pair(item["doc"], item["gen"], item["ref"], item["key"])
        results.append(result)
        
        # Display short CLI preview of the benchmarking output
        print(
            f"{item['doc']} | rows={result['rows_compared']} | "
            f"cos={result['text_cosine_avg']} | exact={result['exact_match_rate']} | "
            f"mae={result['numeric_mae']} | overall={result['overall_score']}"
        )

    if not results:
        print("No evaluation results generated.")
        return

    # Once metrics are stored, dump them into a final evaluation CSV file internally used by developers
    out_path = Path("./generated_outputs/evaluation_doc1_doc3.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Evaluation saved → {out_path}")