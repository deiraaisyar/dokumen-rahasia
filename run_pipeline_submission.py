"""
run_pipeline_submission.py
Master Orchestrator - OECD NEA Coding Competition
[THE ULTIMATE MERGE: Deira's Logic + Hybrid AI Engine]
"""
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 1. Ensure submission_pipeline/ folder is read by Python to allow module imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) # Get the absolute path of the current directory (project root)
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "submission_pipeline") # Create the path to the 'submission_pipeline' directory
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR) # Add the pipeline directory to sys.path so Python can find the custom modules

# 2. Import our merged modules containing the core logic
from extracting import extract_excel, extract_pdf, preprocessing, save_extracted, format_df_to_llm_text
from generating_outputs import process_single_risk, save_cache_to_disk
from save_results import format_and_save_final_excel
from evaluating import run_evaluating

def main():
    # --- PHASE 1: Data Ingestion and Cleaning ---
    print("="*70)
    print("🚀 PHASE 1: EXTRACTION & PREPROCESSING (DEIRA + AI VISION)")
    print("="*70)
    
    try:
        # Run composite data extraction engine to read raw files from the inputs folder
        df1, df2, df3, df4 = extract_excel() # Extracts data from Excel files using both rule-based and AI approaches
        df5 = extract_pdf() # Extracts data from the PDF file using coordinate-based logic
        
        # Run data cleaner on each extracted DataFrame to handle missing values and formatting issues
        df1 = preprocessing(df1, "df1") 
        df2 = preprocessing(df2, "df2")
        df3 = preprocessing(df3, "df3")
        df4 = preprocessing(df4, "df4")
        df5 = preprocessing(df5, "df5")
        
        # Save Audit CSV (Deira's SOP) - Stores the cleaned data as intermediate CSV files for inspection
        save_extracted([df1, df2, df3, df4, df5])
        print("✅ Extraction & Preprocessing Success!")
    except Exception as e:
        # Halt the pipeline if critical extraction fails
        print(f"❌ Failed in Extraction Phase: {e}")
        return

    # --- PHASE 2: AI Processing and Data Transformation ---
    print("\n" + "="*70)
    print("🧠 PHASE 2: AI PREDICTION (HYBRID TRANSFORMER)")
    print("="*70)
    
    OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs" # Define the final output directory
    OUTPUT_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist
    
    # Map each cleaned Dataframe to its corresponding desired Judge's Output Excel Name
    dokumen_map = {
        "1. IVC DOE R2": df1,
        "2. City of York Council": df2,
        "3. Digital Security IT Sample Register": df3,
        "4. Moorgate Crossrail Register": df4,
        "5. Corporate Risk Register": df5
    }
    
    # Iterate through each mapped document to apply AI predictions
    for doc_name, df_clean in dokumen_map.items():
        if df_clean is None or df_clean.empty:
            print(f"⚠️ Skip {doc_name}: Data is empty.") # Handle undefined or failed extraction cases
            continue
            
        print(f"\n🚀 Processing AI for: {doc_name}")
        
        # Convert the tabular DataFrame into a pipe-separated string format suitable for the LLM Prompt
        llm_texts = format_df_to_llm_text(df_clean)
        
        final_results = []
        # Process each row (risk entry) individually utilizing a progress bar (tqdm)
        for text in tqdm(llm_texts, desc=f"AI Inference {doc_name[:15]}"):
            # Prediction Process: Send the formatted text and document context to the AI model
            res = process_single_risk(target_text=text, project_name=doc_name)
            final_results.append(res) # Store the JSON response from the AI
            
        final_excel_name = f"{doc_name} (Final).xlsx" # Construct the final file name
        final_excel_path = OUTPUT_DIR / final_excel_name # Construct the full save path
        
        try:
            # Filter and format the raw AI JSON results into the strict requested Excel structure using save_results.py
            format_and_save_final_excel(final_results, str(final_excel_path))
            print(f"✅ Successfully saved: {final_excel_name}")
        except Exception as e:
            # Fallback: If strict formatting fails, perform a raw dump using Pandas directly
            print(f"⚠️ Failed strict formatting, using Pandas fallback: {e}")
            pd.DataFrame(final_results).to_excel(str(final_excel_path), index=False)
            print(f"✅ Successfully saved (Fallback): {final_excel_name}")
            
    # Save the accumulated LLM responses (AI memory) to a JSON file to serve as a cache for future runs
    try: save_cache_to_disk()
    except: pass
    
    # --- PHASE 3: Benchmark and Evaluation ---
    print("\n" + "="*70)
    print("📊 PHASE 3: AI PERFORMANCE EVALUATION (BENCHMARKING)")
    print("="*70)
    try:
        # Run evaluation metrics (e.g., Cosine Similarity) comparing generated outputs against ground-truth references
        run_evaluating()
    except Exception as e:
        # Treat evaluation errors as non-blocking warnings since the primary outputs are already saved
        print(f"⚠️ Evaluation Report skipped due to minor error: {e}")
        print("💡 Don't panic, Final Excel for Judges is still safe in outputs folder!")
        
    print("\n🏆 ENTIRE PIPELINE COMPLETED! READY FOR SUBMISSION! 🏆")

if __name__ == "__main__":
    main()