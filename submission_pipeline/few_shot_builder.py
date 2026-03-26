"""
src/few_shot_builder.py
-----------------------
Modul Dynamic Few-Shot Learning.
Sistem membaca baris input dan memetakannya ke expected output juri.
"""

import os
import json
import pandas as pd
import threading
from extracting import extract_excel_data, format_df_to_llm_text

FILE_PAIRS = [
    {
        "input": "1. IVC DOE R2 (Input).xlsx",
        "output": "1. IVC DOE (Final).xlsx",
        "mapping": {
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
            "Risk ID": "Risk ID", "Risk Description": "Risk Description",
            "Project Stage": "Project Stage", "Project Category": "Risk Category", # Beda nama kolom
            "Risk Owner": "Risk Owner", "Mitigating Action": "Mitigation",
            "Likelihood (1-10)": "Likelihood (1-10)", "Impact (1-10)": "Impact (1-10)"
        }
    }
]

_CACHE_EXAMPLES_BY_COL = None
_CACHE_LOCK = threading.Lock()

def _load_all_data():
    global _CACHE_EXAMPLES_BY_COL
    with _CACHE_LOCK:
        if _CACHE_EXAMPLES_BY_COL is not None:
            return _CACHE_EXAMPLES_BY_COL
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "inputs"))
        output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "outputs"))
        
        all_columns = FILE_PAIRS[0]["mapping"].keys()
        _CACHE_EXAMPLES_BY_COL = {col: [] for col in all_columns}
        
        print("🧠 [Few-Shot Builder] Mempelajari gaya bahasa Juri dari Golden Data...")
        for pair in FILE_PAIRS:
            in_path = os.path.join(input_dir, pair['input'])
            out_path = os.path.join(output_dir, pair['output'])
            
            if not os.path.exists(in_path) or not os.path.exists(out_path): continue
                
            df_in = extract_excel_data(in_path)
            if df_in is None: continue
            input_texts = format_df_to_llm_text(df_in)
            
            try: df_out = pd.read_excel(out_path)
            except Exception: continue
                
            min_len = min(len(input_texts), len(df_out))
            
            # Ambil maksimal 3 contoh per file agar AI tidak kehabisan kuota token
            for i in range(min(min_len, 3)):
                in_text = input_texts[i]
                if len(in_text) > 400: in_text = in_text[:400] + "... [TRUNCATED]"
                
                for standard_col, actual_col in pair["mapping"].items():
                    if actual_col not in df_out.columns: continue
                        
                    out_val = df_out.iloc[i].get(actual_col, "")
                    if pd.isna(out_val): out_val = ""
                    else:
                        out_val = str(out_val).strip()
                        if standard_col in ["Likelihood (1-10)", "Impact (1-10)"] and out_val.endswith('.0'):
                            out_val = out_val[:-2]
                    
                    if out_val and out_val.lower() not in ['nan', 'none']:
                        _CACHE_EXAMPLES_BY_COL[standard_col].append({
                            "input_text": in_text,
                            "expected_output": out_val
                        })
                        
        return _CACHE_EXAMPLES_BY_COL

def get_few_shots_for_column(col_name):
    """Menghasilkan string JSON untuk diinjeksikan ke otak AI."""
    examples_by_col = _load_all_data()
    if col_name not in examples_by_col or not examples_by_col[col_name]: return "[]"
    # Berikan maksimal 3 contoh per kolom (terbaik untuk LLM)
    return json.dumps(examples_by_col[col_name][:3], ensure_ascii=False, indent=2)