"""
submission_pipeline/extracting.py
---------------------------------
Enterprise-Grade Data Ingestion Layer
[THE MASTER MERGE: AI VISION + DEIRA'S BUSINESS LOGIC]
Combines the strength of Deira's audit structure & Regex 
with Anti-Hardcode Radar & AI Vision.
"""

import os
import re
import json
import warnings
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber

# Hide default pandas/openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.worksheet._reader")
pd.set_option('future.no_silent_downcasting', True)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("⚠️ PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")

def resolve_existing_dir(candidates: list[str], label: str) -> Path:
    """Deira's function: Find a valid directory."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_dir():
            return path
    fallback = Path(candidates[0])
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

INPUT_DIR = resolve_existing_dir(["./inputs", "../inputs"], "inputs")
OUTPUT_DIR = resolve_existing_dir(["./outputs", "../outputs"], "outputs")

# Setup LLM Client (DeepSeek) for PDF Vision
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None
model_name = "deepseek-chat"

# ==============================================================================
# 1. SMART EXCEL EXTRACTOR (ANTI-HARDCODE AI RADAR)
# ==============================================================================
def smart_extract_excel(filepath, header_row_count=1):
    """Detect tables dynamically to be resilient against row/column shifts."""
    try:
        # Attempt to open the Excel file
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        # Log error if file cannot be opened and return empty DataFrame
        print(f"❌ Error opening Excel file {filepath}: {e}")
        return pd.DataFrame()

    # Initialize variables to track the best sheet and header row found
    best_sheet, best_header_row_idx, highest_score = None, -1, -1
    # Define keywords to look for in potential header rows
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 'owner', 'action', 'category', 'status', 'mitigation', 'severity', 'id', 'ref', 'number']
    
    # Iterate through all sheets in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the first 50 rows of the sheet to search for headers
        df_tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None).head(50)
        # Skip empty sheets
        if df_tmp.empty: continue
            
        # Iterate through each row in the sample
        for idx, row in df_tmp.iterrows():
            # Get cells that are not null and not empty strings
            non_null_cells = [val for val in row.values if pd.notna(val) and str(val).strip() != '']
            fill_count = len(non_null_cells)
            # Skip rows with fewer than 3 filled cells (unlikely to be a header)
            if fill_count < 3: continue
                
            keyword_matches = 0
            # Filter for string cells to check against keywords
            string_cells = [val for val in non_null_cells if isinstance(val, str)]
            for val in string_cells:
                # Ignore very long strings (likely data, not headers)
                if len(val) > 60: continue
                cell_lower = val.lower()
                # Count how many keywords are present in this cell
                for kw in keywords:
                    if kw in cell_lower: keyword_matches += 1
                        
            # Calculate a score based on fill count and keyword matches
            score = fill_count + (keyword_matches * 50)  
            # Bonus score if all filled cells are strings
            if len(string_cells) == fill_count: score += 10
            # Penalize rows that are further down (headers are usually near the top)
            score -= idx 
            
            # Update the best sheet and row if this row has the highest score so far
            if score > highest_score:
                highest_score = score
                best_sheet = sheet_name
                best_header_row_idx = idx

    # If no row scored high enough, fallback to basic whole-sheet extraction
    if highest_score < 10:
        return pd.read_excel(filepath)

    # Read the full best sheet
    df_full = pd.read_excel(xls, sheet_name=best_sheet, header=None)
    # Extract the header rows based on the detected index and count
    df_header = df_full.iloc[best_header_row_idx : best_header_row_idx + header_row_count].copy()
    # Extract the data rows below the headers
    df_data = df_full.iloc[best_header_row_idx + header_row_count:].copy()
    
    # Forward-fill merged header cells horizontally
    header_vals = df_header.values.tolist()
    for row_idx in range(len(header_vals)):
        last_val = None
        for col_idx in range(len(header_vals[row_idx])):
            val = header_vals[row_idx][col_idx]
            # Replace empty/null header cells with the last seen value (merged cells)
            if pd.isna(val) or str(val).strip() == '': header_vals[row_idx][col_idx] = last_val
            else: last_val = val
                
    # Reconstruct header dataframe and build final column names
    df_header = pd.DataFrame(header_vals)
    new_headers = []
    for col in df_header.columns:
        # Collect non-empty components for this column's header
        components = [str(val).strip() for val in df_header[col].values if pd.notna(val) and str(val).strip() != '']
        clean_components = []
        # Remove consecutive duplicate components
        for k in components:
            if not clean_components or clean_components[-1] != k: clean_components.append(k)
        # Join components with underscore, or use a default name
        new_headers.append("_".join(clean_components) if clean_components else f"Column_{col}")
        
    # Apply the new headers to the data
    df_data.columns = new_headers
    # Drop rows and columns that are completely empty
    df_data = df_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Filter out invalid rows (e.g., rows with too little data)
    valid_rows = []
    for idx, row in df_data.iterrows():
        # Count cells that have meaningful text
        text_cols = sum(1 for val in row.values if pd.notna(val) and str(val).strip() not in ['0.0', 'NaN', 'None', ''])
        # Consider a row valid if it has at least 2 text columns
        valid_rows.append(text_cols >= 2) 
        
    # Return the cleaned, filtered, and deduplicated dataframe
    return df_data[valid_rows].drop_duplicates().reset_index(drop=True)

# ==============================================================================
# 2. COMBINATION OF EXCEL EXTRACTION (DEIRA'S DOC1 + SMART EXTRACTOR)
# ==============================================================================
def extract_excel():
    print("📊 Extracting Document 1 (Using Deira's Budget Period Logic)...")
    # Load document 1 without headers to process row by row manually
    df_raw = pd.read_excel(INPUT_DIR / '1. IVC DOE R2 (Input).xlsx', sheet_name="Risk Register", header=None)
    # Define exact hardcoded target columns for document 1
    columns = [
        "Revision Date", "RBS Level 1", "RBS Level 2", "Risk Name", "TRL", "TPL",
        "Technology Life Phase", "Risk Owner", "Baseline +/-", "Baseline TYP", "Baseline SEV",
        "Baseline FRQ", "Baseline RPN", "Baseline Description", "Response Strategy",
        "Response Description", "Response Timing", "Residual SEV", "Residual FRQ",
        "Residual RPN", "Residual Description", "Secondary Risks", "Recommendations & Action Items",
        "Contingency Plan",
    ]
    records = []
    current_budget_period = None
    
    # Iterate through all rows in Document 1 to track grouping
    for i, row in df_raw.iterrows():
        val = str(row[0]).strip()
        # Skip the first 4 metadata rows
        if i < 4: continue
        # Detect group headers and update current budget period state
        if val.startswith("Budget Period"):
            current_budget_period = val
            continue
        # Skip fully empty rows
        if row.isna().all(): continue
        # Append data row with its corresponding budget period grouping
        records.append(list(row.values) + [current_budget_period])

    # Construct the final dataframe for document 1
    df1 = pd.DataFrame(records, columns=columns + ["Budget Period"])
    # Reorder columns to place Budget Period at the front
    df1 = df1[["Budget Period"] + [c for c in df1.columns if c != "Budget Period"]].reset_index(drop=True)

    print("📊 Extracting Document 2, 3, 4 (Using AI Density Radar)...")
    # Dynamically extract subsequent Excel files dealing with varied structures
    df2 = smart_extract_excel(INPUT_DIR / '2. City of York Council (Input).xlsx')
    df3 = smart_extract_excel(INPUT_DIR / '3. Digital Security IT Sample Register (Input).xlsx')
    df4 = smart_extract_excel(INPUT_DIR / '4. Moorgate Crossrail Register (Input).xlsx')

    return df1, df2, df3, df4

# ==============================================================================
# 3. AI-DRIVEN PDF EXTRACTOR (REPLACING PDFPLUMBER HARDCODE)
# ==============================================================================
def raw_pdf_to_json(page_text):
    if not client: return []
    system_prompt = """You are a highly precise data parsing assistant.
The following is vertically extracted text from a Corporate Risk Register PDF.
Groups of values reading down the page represent columns.
Output exactly and ONLY a valid JSON array of objects representing these risks.
Keys to extract: Reference, Risk_and_Effects, Mitigation, Risk_Owner, Actions_Being_Taken, Risk_if_No_Action_Likelihood, Risk_if_No_Action_Impact, Current_Risk_Likelihood, Current_Risk_Impact.
Do not wrap your response in markdown fences."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"--- PAGE TEXT ---\n{page_text}"}],
            temperature=0.0
        )
        res = response.choices[0].message.content.strip()
        if res.startswith("```json"): res = res[7:]
        if res.startswith("```"): res = res[3:]
        if res.endswith("```"): res = res[:-3]
        return json.loads(res.strip())
    except Exception: return []

def extract_pdf():
    print("📄 Extracting PDF Document 5 with PdfPlumber (Deira's Accurate Method)...")
    
    # Define exact horizontal boundaries (x0, x1) for each column in the PDF table
    COL_BOUNDS = {
        'ref': (26, 52), 'risk_text': (52, 138), 'impact1': (138, 162),
        'likeli1': (162, 185), 'score1': (185, 210), 'mitigation': (210, 300),
        'impact2': (300, 323), 'likeli2': (323, 347), 'score2': (347, 372),
        'owner': (372, 430), 'actions': (430, 575), 'comments': (575, 850),
    }

    # Helper function to reconstruct text from individual extracted word coordinates
    def words_to_text(wds):
        if not wds: return ''
        # Filter out stray bullet points or special characters
        wds = [w for w in wds if not re.match(r'^[\uf0b7\u2022\uf0a7]$', w['text'])]
        if not wds: return ''
        # Sort words top-to-bottom, then left-to-right (chunking by ~6 pixels vertical)
        wds_sorted = sorted(wds, key=lambda w: (round(w['top'] / 6), w['x0']))
        lines = {}
        # Group words into physical text lines
        for w in wds_sorted:
            key = round(w['top'] / 6)
            lines.setdefault(key, []).append(w['text'])
        # Join words into lines, then lines into a single block of text
        return ' '.join(' '.join(v) for v in lines.values()).strip()

    records = []
    pdf_path = INPUT_DIR / '5. Corporate_Risk_Register (Input).pdf'
    
    # Fail gracefully if Document 5 is missing
    if not pdf_path.exists():
        print(f"⚠️ PDF File not found at: {pdf_path}")
        return pd.DataFrame()

    # Open the PDF for coordinate-based text extraction
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through pages (limit to first 20 just in case)
        for page_num, page in enumerate(pdf.pages[:20]):
            words = page.extract_words(x_tolerance=4, y_tolerance=4)
            page_h = page.height

            # Identify Reference numbers (1-2 digits) residing in the first exact left-column boundary
            ref_words = sorted(
                [w for w in words if re.match(r'^\d{1,2}$', w['text']) and 26 <= w['x0'] <= 35],
                key=lambda w: w['top']
            )

            # Skip the page if no reference numbers exist (no table data here)
            if not ref_words: continue

            # For each reference number, establish vertical chunk bounds based on the next reference number
            for idx, rw in enumerate(ref_words):
                y_start = rw['top'] - 8
                y_end = ref_words[idx + 1]['top'] - 8 if idx + 1 < len(ref_words) else page_h - 30

                row = {'Reference': int(rw['text'])}

                # Extract content chunk by chunk using column boundary boxes
                for col, (x0, x1) in COL_BOUNDS.items():
                    if col == 'ref': continue
                    # Collect all words falling within this cell's bounding box
                    band = [
                        w for w in words
                        if x0 <= w['x0'] < x1 and y_start <= w['top'] <= y_end
                    ]
                    # Form text and attach to row
                    row[col] = words_to_text(band)

                records.append(row)

    # Convert records to Pandas DataFrame and map column names to standard format
    df5 = pd.DataFrame(records).rename(columns={
        'risk_text': 'Risk_and_Effects', 'impact1': 'Risk_if_No_Action_Impact',
        'likeli1': 'Risk_if_No_Action_Likelihood', 'score1': 'Risk_if_No_Action_Score',
        'mitigation': 'Mitigation', 'impact2': 'Current_Risk_Impact',
        'likeli2': 'Current_Risk_Likelihood', 'score2': 'Current_Risk_Score',
        'owner': 'Risk_Owner', 'actions': 'Actions_Being_Taken', 'comments': 'Comments_and_Progress',
    })

    # Data Post-processing: sort numeric References, drop invalid index
    if not df5.empty:
        df5 = df5.sort_values('Reference').reset_index(drop=True)
        # Coerce numeric values in score, impact, and likelihood columns
        for col in ['Risk_if_No_Action_Impact', 'Risk_if_No_Action_Likelihood', 'Risk_if_No_Action_Score',
                    'Current_Risk_Impact', 'Current_Risk_Likelihood', 'Current_Risk_Score']:
            if col in df5.columns:
                df5[col] = pd.to_numeric(df5[col], errors='coerce')

    return df5

# ==============================================================================
# 4. DEIRA'S REGEX SPLITTER & PREPROCESSING LOGIC
# ==============================================================================
def split_risk_effects(df):
    target_col = next((col for col in df.columns if "risk" in col.lower() and "effect" in col.lower()), None)
    if not target_col: return df

    risks, effects = [], []
    for text in df[target_col].fillna(""):
        text = str(text)
        risk_match = re.search(r"risk[:\-]\s*(.*?)\s*effects?[:\-]", text, re.IGNORECASE)
        effect_match = re.search(r"effects?[:\-]\s*(.*)", text, re.IGNORECASE)
        risks.append(risk_match.group(1).strip() if risk_match else text)
        effects.append(effect_match.group(1).strip() if effect_match else "")

    insert_loc = df.columns.get_loc(target_col)
    df.insert(insert_loc, "Risk", risks)
    df.insert(insert_loc + 1, "Effects", effects)
    df = df.drop(columns=[target_col])
    return df

def preprocessing(df, name):
    if df.empty: return df
    df = df.dropna(thresh=max(1, df.shape[1] - 5))
    if name == "df5":
        df = split_risk_effects(df)
    return df

# ==============================================================================
# 5. DEIRA'S AUDIT LOGGING (Save to CSV)
# ==============================================================================
def save_extracted(dfs):
    output_dir = Path("./extracted_inputs")
    output_dir.mkdir(exist_ok=True)
    names = ["df1", "df2", "df3", "df4", "df5"]
    for df, name in zip(dfs, names):
        if df is not None and not df.empty:
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)

def extract_from_outputs_folder():
    input_dir = OUTPUT_DIR
    output_dir = Path("./extracted_outputs")
    output_dir.mkdir(exist_ok=True)
    xlsx_files = list(input_dir.glob("*.xlsx"))

    if not xlsx_files:
        print("no xlsx files found in outputs folder")
        return

    for file in xlsx_files:
        print(f"processing {file.name}")
        try: df = pd.read_excel(file)
        except Exception as e:
            print(f"failed to read {file.name}: {e}")
            continue
        output_path = output_dir / f"{file.stem}.csv"
        df.to_csv(output_path, index=False)

# ==============================================================================
# 6. LLM TEXT FORMATTER (BRIDGE TO MAIN PIPELINE)
# ==============================================================================
def format_df_to_llm_text(df):
    if df is None or df.empty: return []
    llm_texts = []
    for idx, row in df.iterrows():
        row_texts = []
        for col_name, val in row.items():
            clean_val = str(val).replace('\n', ' ').replace('\r', ' ').strip()
            if pd.notna(val) and clean_val != '' and clean_val.lower() not in ['nan', 'none']:
                row_texts.append(f"{col_name}: {clean_val}")
        llm_texts.append(" | ".join(row_texts))
    return llm_texts

def main():
    df1, df2, df3, df4 = extract_excel()
    df5 = extract_pdf()

    df1 = preprocessing(df1, "df1")
    df2 = preprocessing(df2, "df2")
    df3 = preprocessing(df3, "df3")
    df4 = preprocessing(df4, "df4")
    df5 = preprocessing(df5, "df5")

    save_extracted([df1, df2, df3, df4, df5])
    extract_from_outputs_folder()
    print("✅ Extraction complete, files saved!")

if __name__ == "__main__":
    main()