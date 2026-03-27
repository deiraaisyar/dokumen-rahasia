"""
submission_pipeline/generating_outputs.py
-----------------------------------------
The Ultimate Hybrid Engine: Flat JSON & Smart Fallback Logic.
Uses original data values strictly. Math conversions are only applied 
if the original document is completely missing the data.
"""

import os
import re
import json
import hashlib
import threading
import difflib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import pandas as pd

try:
    from few_shot_builder import get_few_shots_for_column
except ImportError:
    def get_few_shots_for_column(col): return ""

# ==============================================================================
# 0. SETUP API (KOBOILLM)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv())

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.koboillm.com/v1")
model_name = os.getenv("DEEPSEEK_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

# ==============================================================================
# 1. OPTIMIZED CACHING SYSTEM (Nama file diganti agar mulai dari 0)
# ==============================================================================
CACHE_DIR = os.path.join(BASE_DIR, "..", "generated_ouputs", "debug_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "llm_clean_cache.json")
LLM_CACHE = {}
CACHE_LOCK = threading.Lock()
CACHE_MODIFIED = False 

def load_cache():
    global LLM_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f: LLM_CACHE = json.load(f)
        except Exception: LLM_CACHE = {}
load_cache()

def save_cache_to_disk():
    global CACHE_MODIFIED
    if not CACHE_MODIFIED: return
    with CACHE_LOCK:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(LLM_CACHE, f, ensure_ascii=False, separators=(',', ':'))
            CACHE_MODIFIED = False
        except OSError: pass

def get_cache_key(prompt, text):
    return hashlib.md5(f"{model_name}_{prompt}_{text}".encode('utf-8')).hexdigest()

# ==============================================================================
# 2. STRICT GOLDEN SET & TEXT FORMATTER
# ==============================================================================
def extract_golden_sets():
    base_ref = Path(BASE_DIR).parent / "preprocessed_outpus"
    stages, categories, owners = set(), set(), set()
    def process_df(df):
        for c in df.columns:
            c_low = str(c).lower()
            if "stage" in c_low or "life" in c_low: stages.update(df[c].dropna().astype(str).unique())
            if "category" in c_low or "rbs" in c_low: categories.update(df[c].dropna().astype(str).unique())
            if "owner" in c_low: owners.update(df[c].dropna().astype(str).unique())

    if base_ref.exists():
        for file in base_ref.glob("*.*"):
            if file.name.startswith(("1", "2", "3")):
                try: 
                    if file.suffix == '.xlsx': process_df(pd.read_excel(file))
                    else: process_df(pd.read_csv(file))
                except Exception: continue

    stg = {s.strip().title() for s in stages if len(str(s).strip()) > 2 and str(s).lower() not in ['nan', 'none', 'na']}
    cat = {c.strip().title() for c in categories if len(str(c).strip()) > 2 and str(c).lower() not in ['nan', 'none', 'na']}
    own = set()
    for o in owners:
        o_str = str(o).strip()
        if len(o_str) > 2 and o_str.lower() not in ['nan', 'none', 'na']:
            match = re.search(r'\((.*?)\)', o_str)
            own.add(match.group(1).title() if match else o_str.title())

    if not stg: stg = {"Pre-Construction", "Construction", "Operational", "Design"}
    if not cat: cat = {"Technical", "Management", "Commercial", "External", "Financial"}
    if not own: own = {"Project Manager", "Lead Engineer", "Environmental", "It Manager"}
    return list(stg), list(cat), list(own)

VALID_STAGES, VALID_CATEGORIES, VALID_OWNERS = extract_golden_sets()

def force_exact_match(val, valid_list, fallback="Unknown"):
    val_lower = str(val).strip().lower()
    if not val_lower or val_lower in ['none', 'null', 'unknown', 'na', 'n/a']: return fallback
    for opt in valid_list:
        if opt.lower() == val_lower: return opt
    matches = difflib.get_close_matches(val_lower, [v.lower() for v in valid_list], n=1, cutoff=0.35)
    if matches: return next(v for v in valid_list if v.lower() == matches[0])
    return fallback

def capitalize_each_word(value):
    if not isinstance(value, str): return value
    text = value.strip()
    if not text: return text
    def _convert_word(match):
        word = match.group(0)
        if word.isupper() and len(word) <= 4: return word
        return word[0].upper() + word[1:].lower()
    converted = re.sub(r"[A-Za-z][A-Za-z'/-]*", _convert_word, text)
    first_alpha = re.search(r"[A-Za-z]", converted)
    if first_alpha:
        i = first_alpha.start()
        converted = converted[:i] + converted[i].upper() + converted[i + 1:]
    return converted

# ==============================================================================
# 3. SMART DATA EXTRACTION & TEXT-TO-SCALE CONVERTER
# ==============================================================================
def extract_explicit_values(target_text):
    """Membaca data secara spesifik dari format Key: Value menggunakan String Parsing"""
    explicit = {}
    
    # Split text berdasarkan separator '|'
    pairs = [p.strip() for p in target_text.split('|')]
    for pair in pairs:
        if ':' in pair:
            k, v = pair.split(':', 1)
            k_lower = k.strip().lower()
            v_clean = v.strip()
            
            if not v_clean or v_clean.lower() in ['nan', 'none', 'null', '']:
                continue
                
            # Mapping pintar untuk berbagai variasi nama kolom/key di raw data
            if 'risk_if_no_action_likelihood' in k_lower or 'likelihood1' in k_lower:
                explicit['Likelihood_No_Action'] = v_clean
            elif 'risk_if_no_action_impact' in k_lower or 'impact1' in k_lower:
                explicit['Impact_No_Action'] = v_clean
            elif 'current_risk_likelihood' in k_lower or 'likelihood2' in k_lower or 'residual frq' in k_lower or 'post likelihood' in k_lower:
                explicit['Current_Likelihood'] = v_clean
            elif 'current_risk_impact' in k_lower or 'impact2' in k_lower or 'residual sev' in k_lower or 'post impact' in k_lower:
                explicit['Current_Impact'] = v_clean
            elif 'baseline frq' in k_lower or ('likelihood' in k_lower and 'no_action' not in k_lower and 'current' not in k_lower):
                explicit['Likelihood'] = v_clean
            elif 'baseline sev' in k_lower or ('impact' in k_lower and 'no_action' not in k_lower and 'current' not in k_lower):
                explicit['Impact'] = v_clean
            elif 'project stage' in k_lower or 'life phase' in k_lower:
                explicit['Project Stage'] = v_clean
            elif 'project category' in k_lower or 'rbs' in k_lower:
                explicit['Project Category'] = v_clean
            elif 'owner' in k_lower:
                explicit['Risk Owner'] = v_clean
                
    return explicit

def text_to_score(val_str, max_scale):
    """
    Mengonversi teks kualitatif (High/Med/Low) atau angka murni menjadi integer.
    Jika kosong atau tidak dikenali, return None agar fungsi utama tahu harus menghitung manual.
    """
    if val_str is None: return None
    text = str(val_str).strip().lower()
    
    # 1. Coba ekstrak angka jika ada
    nums = re.findall(r'\d+', text)
    if nums:
        try:
            return max(1, min(max_scale, int(nums[0])))
        except: pass
            
    # 2. Coba mapping kata
    if max_scale == 5:
        if any(w in text for w in ['high', 'very high', 'critical', 'almost certain', 'severe']): return 5
        if any(w in text for w in ['medium', 'med', 'possible', 'moderate', 'likely']): return 3
        if any(w in text for w in ['low', 'very low', 'rare', 'minor', 'unlikely', 'trivial']): return 1
    else: # Skala 10
        if any(w in text for w in ['high', 'very high', 'critical', 'almost certain', 'severe']): return 9
        if any(w in text for w in ['medium', 'med', 'possible', 'moderate', 'likely', 'serious']): return 5
        if any(w in text for w in ['low', 'very low', 'rare', 'minor', 'unlikely', 'trivial']): return 2
        
    return None

def calc_priority_doc5_scale_1_5(l, i):
    try:
        score = float(l) * float(i)
        if score <= 5: return "Low"
        elif score <= 14: return "Med"
        else: return "High"
    except Exception: return "Med"

def calculate_priority_math(l, i):
    try:
        score = float(l) * float(i)
        if score <= 20: return "Low"
        elif score <= 50: return "Med"
        else: return "High"
    except Exception: return "Med"

# ==============================================================================
# 4. AI ENGINE (FLAT JSON, NO REASONING - SUPER CEPAT)
# ==============================================================================
def process_single_risk(target_text, project_name=""):
    global CACHE_MODIFIED
    explicit_data = extract_explicit_values(target_text)
    
    try:
        sample_desc = get_few_shots_for_column("Risk Description")
        sample_mitigation = get_few_shots_for_column("Mitigating Action")
    except:
        sample_desc, sample_mitigation = "", ""
        
    doc_backgrounds = {
        "IVC": "Igiugig Village Council (IVC) MHK project.",
        "York": "City of York Council construction project.",
        "Digital": "Digital security and IT risk register.",
        "Moorgate": "Moorgate Crossrail construction project.",
        "Corporate": "Corporate risk register. Operations, HR, IT. NOT a construction project."
    }
    
    current_bg = "A general project."
    for key, bg in doc_backgrounds.items():
        if key.lower() in project_name.lower():
            current_bg = bg; break
        
    # 🔥 PROMPT BERSIH: JSON Dibuat Rata, Tanpa Reasoning! AI Memproses 2x Lebih Cepat 🔥
    system_prompt = f"""You are an elite LLM Data Parsing Engine.
[CONTEXT] Project Name: {project_name} | Background: {current_bg}

[CONSTRAINTS]
- Project_Category MUST BE from: {json.dumps(VALID_CATEGORIES)}
- Project_Stage MUST BE from: {json.dumps(VALID_STAGES)}
- Risk_Owner MUST BE from: {json.dumps(VALID_OWNERS)}

[FEW-SHOT EXAMPLES]
- Risk Description Style: {sample_desc[:150]}
- Mitigating Action Style: {sample_mitigation[:150]}

[OUTPUT FORMAT]
OUTPUT EXACTLY ONE FLAT JSON OBJECT. DO NOT ADD ANY REASONING FIELDS.
{{
    "Risk_ID": "...",
    "Risk_Description": "...",
    "Project_Category": "...",
    "Risk_Owner": "...",
    "Project_Stage": "...",
    "Mitigating_Action": "...",
    "Likelihood": "exact number or text",
    "Impact": "exact number or text",
    "Current_Likelihood": "exact number or text",
    "Current_Impact": "exact number or text"
}}"""

    user_payload = f"--- RAW ROW DATA ({project_name}) ---\n{target_text}"
    cache_key = get_cache_key(system_prompt, user_payload)
    parsed_json = {}
    
    with CACHE_LOCK:
        if cache_key in LLM_CACHE: parsed_json = LLM_CACHE[cache_key]

    if not parsed_json and client:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
                temperature=0.0
            )
            raw_ans = re.sub(r"^```json\s*|^```\s*|\s*```$", "", response.choices[0].message.content.strip())
            parsed_json = json.loads(raw_ans)
            
            if hasattr(response, 'usage') and response.usage:
                log_api_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                
            with CACHE_LOCK:
                LLM_CACHE[cache_key] = parsed_json
                CACHE_MODIFIED = True 
        except Exception: pass

    return _post_process_hybrid(parsed_json, explicit_data, project_name)

def _get_val(obj, key, default=""):
    if not isinstance(obj, dict): return default
    return obj.get(key, default)

def _post_process_hybrid(parsed_json, explicit_data, project_name=""):
    results = {}
    if not isinstance(parsed_json, dict): parsed_json = {}
    
    results["Risk ID"] = parsed_json.get("Risk_ID", "R-UNK")
    results["Risk Description"] = parsed_json.get("Risk_Description", "Unspecified")
    results["Mitigating Action"] = parsed_json.get("Mitigating_Action", "Monitor and evaluate.")
    
    raw_cat = explicit_data.get("Project Category", parsed_json.get("Project_Category", "Technical"))
    raw_own = explicit_data.get("Risk Owner", parsed_json.get("Risk_Owner", "Unknown"))
    raw_stg = explicit_data.get("Project Stage", parsed_json.get("Project_Stage", "Operational"))
    
    cat_final = force_exact_match(raw_cat, VALID_CATEGORIES, "Technical")
    stg_final = force_exact_match(raw_stg, VALID_STAGES, "Operational")
    
    cat_lower = cat_final.lower()
    if cat_lower in ["technical", "design", "quality"]: def_own = "Lead Engineer"
    elif cat_lower in ["financial", "commercial", "management", "procurement", "stakeholder"]: def_own = "Project Manager"
    elif cat_lower == "environmental" or "legis" in cat_lower: def_own = "Environmental"
    elif "it" in cat_lower or "digital" in cat_lower: def_own = "It Manager"
    else: def_own = "Project Manager"
    
    own_final = force_exact_match(raw_own, VALID_OWNERS, def_own)
    
    results["Project Category"] = capitalize_each_word(cat_final)
    results["Risk Owner"] = capitalize_each_word(own_final)
    results["Project Stage"] = capitalize_each_word(stg_final)
    
    is_doc_5 = "corporate" in project_name.lower() or "5" in str(project_name)
    
    # =========================================================================
    # LOGIKA ANGKA CERDAS (PRIORITAS: DATA ASLI -> KONVERSI -> KALKULASI JIKA KOSONG)
    # =========================================================================
    if is_doc_5:
        max_scale = 5
        # 1. NO ACTION (Pre-mitigation)
        raw_l = explicit_data.get("Likelihood_No_Action", parsed_json.get("Likelihood"))
        raw_i = explicit_data.get("Impact_No_Action", parsed_json.get("Impact"))
        
        l_pre = text_to_score(raw_l, max_scale)
        i_pre = text_to_score(raw_i, max_scale)
        
        # Hitung manual HANYA JIKA datanya memang kosong/tidak valid
        if l_pre is None: l_pre = 3 
        if i_pre is None: i_pre = 3
            
        results["Likelihood No Action (1-5)"] = l_pre
        results["Impact No Action (1-5)"] = i_pre
        results["Risk Priority No Action (low, med, high)"] = calc_priority_doc5_scale_1_5(l_pre, i_pre)
        
        # 2. CURRENT (Post-mitigation)
        raw_cur_l = explicit_data.get("Current_Likelihood", parsed_json.get("Current_Likelihood"))
        raw_cur_i = explicit_data.get("Current_Impact", parsed_json.get("Current_Impact"))
        
        l_cur = text_to_score(raw_cur_l, max_scale)
        i_cur = text_to_score(raw_cur_i, max_scale)
        
        # Hitung manual HANYA JIKA data Current Risk-nya memang tidak disebutkan
        if l_cur is None: l_cur = max(1, l_pre - 1)
        if i_cur is None: i_cur = max(1, i_pre - 1)
            
        results["Likelihood Current (1-5)"] = l_cur
        results["Impact Current (1-5)"] = i_cur
        results["Risk Priority Current (low, med, high)"] = calc_priority_doc5_scale_1_5(l_cur, i_cur)
        
    else: 
        max_scale = 10
        raw_l = explicit_data.get("Likelihood", parsed_json.get("Likelihood"))
        raw_i = explicit_data.get("Impact", parsed_json.get("Impact"))
        
        l_pre = text_to_score(raw_l, max_scale)
        i_pre = text_to_score(raw_i, max_scale)
        
        if l_pre is None: l_pre = 5
        if i_pre is None: i_pre = 5
            
        results["Likelihood (1-10) (pre-mitigation)"] = l_pre
        results["Impact (1-10) (pre-mitigation)"] = i_pre
        results["Risk Priority (pre-mitigation)"] = calculate_priority_math(l_pre, i_pre)
        
        raw_cur_l = explicit_data.get("Current_Likelihood", parsed_json.get("Current_Likelihood"))
        raw_cur_i = explicit_data.get("Current_Impact", parsed_json.get("Current_Impact"))
        
        l_cur = text_to_score(raw_cur_l, max_scale)
        i_cur = text_to_score(raw_cur_i, max_scale)
        
        if l_cur is None: l_cur = max(1, int(l_pre * 0.6))
        if i_cur is None: i_cur = max(1, int(i_pre * 0.8))
        
        results["Likelihood (1-10) (post-mitigation)"] = l_cur
        results["Impact (1-10) (post-mitigation)"] = i_cur
        results["Risk Priority (post-mitigation)"] = calculate_priority_math(l_cur, i_cur)

    return results