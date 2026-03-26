import os
import json
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LIKELIHOOD_MAP = {"rare": 2, "unlikely": 4, "possible": 6, "likely": 8, "almost certain": 10}
IMPACT_MAP     = {"minor": 2, "serious": 5, "major": 8, "critical": 10}

DOC1_CONTEXT = (
    "This is a risk register for the Igiugig Village Council (IVC) Marine and Hydrokinetic (MHK) "
    "river power system project, funded by the US Department of Energy (DOE). "
    "It involves technology development, procurement, deployment, and operation of a river turbine "
    "in a remote Alaskan village."
)

DOC2_CONTEXT = (
    "This is a risk register for a construction and renovation project by City of York Council, UK. "
    "It involves refurbishment of a historic public building, including structural, design, "
    "planning, and procurement risks across pre-construction, construction, and commissioning stages."
)

DOC3_CONTEXT = (
    "This is a digital security and IT risk register for an organisation's internal IT infrastructure. "
    "It covers cybersecurity, backup and recovery, infrastructure resilience, and compliance risks "
    "during ongoing operations."
)

DOC4_CONTEXT = (
    "This is a risk register for the Moorgate Crossrail Street Level (MCSL) project, "
    "a public realm and street improvement construction project in Moorgate, London. "
    "It involves road redesign, stakeholder engagement, and dependency on Crossrail station opening."
)

DOC5_CONTEXT = (
    "This is a corporate risk register for Fenland District Council, "
    "a UK local government body. Risks span IT, finance, governance, HR, and emergency planning. "
    "This is not a construction project — it is an ongoing operational/corporate register."
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Send a prompt to Groq and return the plain text response.
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# Collect valid values from existing outputs to constrain LLM answers.
def load_valid_values() -> tuple:
    """Load valid Project Stage, Category, Risk Owner from existing output files (doc1-3)."""
    data_dir = Path("./data/preprocessed_inputs")
    stage_vals, category_vals, owner_vals = [], [], []

    for file in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(file)
        if "Project Stage" in df.columns:
            stage_vals.extend(df["Project Stage"].dropna().astype(str).unique().tolist())
        if "Project Category" in df.columns:
            category_vals.extend(df["Project Category"].dropna().astype(str).unique().tolist())
        if "Risk Owner" in df.columns:
            owner_vals.extend(df["Risk Owner"].dropna().astype(str).unique().tolist())

    return list(set(stage_vals)), list(set(category_vals)), list(set(owner_vals))

VALID_STAGES, VALID_CATEGORIES, VALID_OWNERS = load_valid_values()

# Safely parse JSON object from an LLM response string.
def parse_json_response(text: str) -> dict:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


# Calculate risk priority for 1-10 style scores.
def calc_priority(likelihood, impact) -> str:
    try:
        score = float(likelihood) * float(impact)
        if score <= 20:   return "Low"
        elif score <= 50: return "Med"
        return "High"
    except Exception:
        return ""


# Calculate risk priority for a 1-5 risk matrix.
def calc_priority_doc5_scale_1_5(likelihood, impact) -> str:
    """Calculate priority for a 1-5 risk matrix (max score 25)."""
    try:
        score = float(likelihood) * float(impact)
        if score <= 5:
            return "Low"
        elif score <= 14:
            return "Med"
        return "High"
    except Exception:
        return ""


# Pick the first existing path from a list of candidate locations.
def resolve_existing_path(candidates: list[str], label: str) -> Path:
    """Return the first existing path from candidates or raise a clear error."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"No available file found for {label}: {candidates}")


# Normalize row identifiers before matching across files.
def normalize_key(value) -> str:
    """Normalize identifier values so cross-file row matching is more stable."""
    if pd.isna(value):
        return ""
    return str(value).strip()


# Build fast lookup by trying key columns in order.
def build_row_lookup(df: pd.DataFrame, key_columns: list[str]) -> dict:
    """Build a lookup dictionary by trying key columns in priority order."""
    lookup = {}
    for _, row in df.iterrows():
        for key_column in key_columns:
            if key_column in df.columns:
                key = normalize_key(row.get(key_column, ""))
                if key:
                    lookup[key] = row
                    break
    return lookup


# Read a value safely from a row-like object.
def get_value(row: pd.Series, column: str, default=""):
    """Safely fetch a value from a row-like object."""
    if row is None:
        return default
    return row.get(column, default)


# Convert revision date into generated output display format.
def format_revision_date(value) -> str:
    """Format dates like 2017-02-21 into 21-Feb-17."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return text
    return dt.strftime("%d-%b-%y")


# Capitalize the first letter of each word for output text values.
def capitalize_each_word(value):
    """Convert text to title case while preserving short all-caps acronyms."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    def _convert_word(match):
        word = match.group(0)
        if word.isupper() and len(word) <= 4:
            return word
        return word[0].upper() + word[1:].lower()

    converted = re.sub(r"[A-Za-z][A-Za-z'/-]*", _convert_word, text)

    # Ensure the first alphabetic character is uppercase.
    first_alpha = re.search(r"[A-Za-z]", converted)
    if first_alpha:
        i = first_alpha.start()
        converted = converted[:i] + converted[i].upper() + converted[i + 1:]

    return converted


# Apply title-case formatting only on selected metadata columns.
def apply_title_case_to_selected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply title-case normalization only to Project Stage/Category and Risk Owner."""
    df_copy = df.copy()
    target_columns = ["Project Stage", "Project Category", "Risk Owner"]
    for col in target_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(capitalize_each_word)
    return df_copy


# Ask LLM to infer the most suitable project stage for doc3 risk.
def infer_project_stage_doc3(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC3_CONTEXT}\n\n"
        f"Assign the most appropriate Project Stage for this IT/cyber risk.\n"
        f"Valid stages (choose ONLY from this list): {', '.join(VALID_STAGES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_stage\": \"Operations\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_stage", "")


# Ask LLM to infer the most suitable project category for doc3 risk.
def infer_project_category_doc3(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC3_CONTEXT}\n\n"
        f"Classify this IT/cyber risk into ONE project category.\n"
        f"Valid categories (choose ONLY from this list): {', '.join(VALID_CATEGORIES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_category\": \"Operational\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_category", "")


# Ask LLM to infer the most suitable risk owner role for doc3 risk.
def infer_risk_owner_doc3(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC3_CONTEXT}\n\n"
        f"Assign the most suitable Risk Owner role for this IT/cyber risk.\n"
        f"Valid roles (choose ONLY from this list): {', '.join(VALID_OWNERS)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"risk_owner\": \"IT Manager\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("risk_owner", "")


# Ask LLM to infer the most suitable project stage for doc2 risk.
def infer_project_stage_doc2(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC2_CONTEXT}\n\n"
        f"Assign the most appropriate Project Stage for this construction risk.\n"
        f"Valid stages (choose ONLY from this list): {', '.join(VALID_STAGES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_stage\": \"Construction\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_stage", "")


# Ask LLM to infer the most suitable project category for doc2 risk.
def infer_project_category_doc2(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC2_CONTEXT}\n\n"
        f"Classify this risk into ONE project category.\n"
        f"Valid categories (choose ONLY from this list): {', '.join(VALID_CATEGORIES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_category\": \"Planning\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_category", "")


# Ask LLM to infer the most suitable project stage for doc1 risk.
def infer_project_stage_doc1(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC1_CONTEXT}\n\n"
        f"Assign the most appropriate Project Stage for this project risk.\n"
        f"Valid stages (choose ONLY from this list): {', '.join(VALID_STAGES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_stage\": \"Construction\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_stage", "")


# Ask LLM to infer the most suitable project category for doc1 risk.
def infer_project_category_doc1(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC1_CONTEXT}\n\n"
        f"Classify this risk into ONE project category.\n"
        f"Valid categories (choose ONLY from this list): {', '.join(VALID_CATEGORIES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_category\": \"Planning\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_category", "")


# ── Doc 4 helpers ──────────────────────────────────────────────────────────────

def infer_project_stage_doc4(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC4_CONTEXT}\n\n"
        f"Choose the most appropriate Project Stage for this construction project risk.\n"
        f"Valid stages (choose ONLY from this list): {', '.join(VALID_STAGES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_stage\": \"Construction\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_stage", "")


def infer_project_category_doc4(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC4_CONTEXT}\n\n"
        f"Classify this risk into ONE project category.\n"
        f"Valid categories (choose ONLY from this list): {', '.join(VALID_CATEGORIES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_category\": \"Stakeholder\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_category", "")


def infer_risk_owner_doc4(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC4_CONTEXT}\n\n"
        f"Assign a job role as Risk Owner for this risk.\n"
        f"Valid roles (choose ONLY from this list): {', '.join(VALID_OWNERS)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"risk_owner\": \"Project Manager\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("risk_owner", "")


# ── Doc 5 helpers ──────────────────────────────────────────────────────────────

def infer_project_stage_doc5(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC5_CONTEXT}\n\n"
        f"Assign a lifecycle stage for this corporate/operational risk.\n"
        f"Valid stages (choose ONLY from this list): {', '.join(VALID_STAGES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_stage\": \"Operations\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_stage", "")


def infer_project_category_doc5(risk_id: str, desc: str) -> str:
    prompt = (
        f"Context: {DOC5_CONTEXT}\n\n"
        f"Classify this risk into ONE category.\n"
        f"Valid categories (choose ONLY from this list): {', '.join(VALID_CATEGORIES)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"project_category\": \"Financial\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("project_category", "")


def infer_risk_owner_doc5(risk_id: str, desc: str, raw_owner: str) -> str:
    prompt = (
        f"Context: {DOC5_CONTEXT}\n\n"
        f'The risk register lists "{raw_owner}" as owner. Convert this to a proper job role title.\n'
        f'"CMT" means Corporate Management Team. Named individuals should become their likely role.\n'
        f"Valid roles (choose ONLY from this list): {', '.join(VALID_OWNERS)}\n\n"
        f"Risk ID: {risk_id}\n"
        f"Risk Description: {desc}\n"
        f"Raw Owner: {raw_owner}\n\n"
        f"Respond ONLY with a JSON object. Example: {{\"risk_owner\": \"IT Manager\"}}"
    )
    return parse_json_response(call_llm(prompt)).get("risk_owner", raw_owner)


# ── Main generators ────────────────────────────────────────────────────────────

# Generate doc1 output using extracted/preprocessed mappings and LLM metadata.
def generate_doc1():
    # Use extracted data for date/description/mitigation and risk scores.
    extracted_path = resolve_existing_path(
        ["./data/extracted_outputs/df1.csv", "./data/extracted_inputs/df1.csv"],
        "doc1 extracted source",
    )

    # Use preprocessed data for Risk Owner.
    preprocessed_path = resolve_existing_path(
        ["./data/preprocessed_outputs/df1.csv", "./data/preprocessed_inputs/df1.csv"],
        "doc1 preprocessed source",
    )

    output_path = Path("./data/generated_outputs/doc1.csv")
    ext_df = pd.read_csv(extracted_path)
    pre_df = pd.read_csv(preprocessed_path)

    rows = []
    for i, ext_row in ext_df.iterrows():
        # Build generated Risk ID manually from row order.
        generated_risk_id = i + 1

        # Use aligned preprocessed row for Risk Owner if available.
        pre_row = pre_df.iloc[i] if i < len(pre_df) else None

        # Map extracted values to generated schema.
        risk_desc = get_value(ext_row, "Baseline Description", "")
        pre_likelihood = get_value(ext_row, "Baseline FRQ", "")
        pre_impact = get_value(ext_row, "Baseline SEV", "")
        post_likelihood = get_value(ext_row, "Residual FRQ", "")
        post_impact = get_value(ext_row, "Residual SEV", "")

        rows.append({
            "Date Added": format_revision_date(get_value(ext_row, "Revision Date", "")),
            "Risk ID": generated_risk_id,
            "Risk Description": risk_desc,
            "Project Stage": infer_project_stage_doc1(generated_risk_id, risk_desc),
            "Project Category": infer_project_category_doc1(generated_risk_id, risk_desc),
            "Risk Owner": get_value(pre_row, "Risk Owner", ""),
            "Likelihood (1-10) (pre-mitigation)": pre_likelihood,
            "Impact (1-10) (pre-mitigation)": pre_impact,
            "Risk Priority (pre-mitigation)": calc_priority(pre_likelihood, pre_impact),
            "Mitigating Action": get_value(ext_row, "Response Description", ""),
            "Likelihood (1-10) (post-mitigation)": post_likelihood,
            "Impact (1-10) (post-mitigation)": post_impact,
            "Risk Priority (post-mitigation)": calc_priority(post_likelihood, post_impact),
        })
        print(f"  doc1 processed: {generated_risk_id}")

    # Apply title-case formatting only to selected metadata columns.
    output_df = apply_title_case_to_selected_columns(pd.DataFrame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Doc1 done → {output_path}")

# Generate doc2 output using requested field mapping and LLM-inferred metadata.
def generate_doc2():
    # Use preprocessed data as the source for LLM understanding.
    preprocessed_path = resolve_existing_path(
        ["./data/preprocessed_outputs/df2.csv", "./data/preprocessed_inputs/df2.csv"],
        "doc2 preprocessed source",
    )

    # Use extracted data for columns that must be copied exactly.
    extracted_path = resolve_existing_path(
        ["./data/extracted_outputs/df2.csv", "./data/extracted_inputs/df2.csv"],
        "doc2 extracted source",
    )

    output_path = Path("./data/generated_outputs/doc2.csv")
    pre_df = pd.read_csv(preprocessed_path)
    ext_df = pd.read_csv(extracted_path)

    # Match rows primarily by Risk ID.
    ext_lookup = build_row_lookup(ext_df, ["Risk ID"])

    rows = []
    for i, pre_row in pre_df.iterrows():
        # Skip accidental header row that may appear in preprocessed data.
        pre_risk_id = get_value(pre_row, "Risk ID", "")
        if str(pre_risk_id).strip().lower() == "risk id":
            continue

        # Build LLM context from preprocessed content.
        risk_id = pre_risk_id
        llm_desc = get_value(pre_row, "Risk Description", "")

        # Resolve extracted row for copied columns.
        ext_row = ext_lookup.get(normalize_key(risk_id))
        if ext_row is None and i < len(ext_df):
            ext_row = ext_df.iloc[i]

        # Copy required columns exactly from extracted data.
        out_risk_id = get_value(ext_row, "Risk ID", risk_id)
        out_desc = get_value(ext_row, "Risk Description", llm_desc)
        likelihood = get_value(ext_row, "Likelihood (1-10)", "")
        impact = get_value(ext_row, "Impact (1-10)", "")
        risk_owner = get_value(ext_row, "Risk Owner", "")

        rows.append({
            "Date Added":                    "",
            "Risk ID":                       out_risk_id,
            "Risk Description":              out_desc,
            "Project Stage":                 infer_project_stage_doc2(out_risk_id, llm_desc),
            "Project Category":              infer_project_category_doc2(out_risk_id, llm_desc),
            "Likelihood (1-10)":             likelihood,
            "Impact (1-10)":                 impact,
            "Risk Priority (low, med, high)": calc_priority(likelihood, impact),
            "Risk Owner":                    risk_owner,
            "Mitigating Action":             get_value(ext_row, "Impact", ""),
            "Result":                        get_value(ext_row, "Mitigation", ""),
        })
        print(f"  doc2 processed: {out_risk_id}")

    # Apply title-case formatting only to selected metadata columns.
    output_df = apply_title_case_to_selected_columns(pd.DataFrame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Doc2 done → {output_path}")

# Generate doc3 output using extracted values + LLM-inferred metadata.
def generate_doc3():
    # Use preprocessed data as the source for LLM understanding.
    preprocessed_path = resolve_existing_path(
        ["./data/preprocessed_outputs/df3.csv", "./data/preprocessed_inputs/df3.csv"],
        "doc3 preprocessed source",
    )

    # Use extracted data for columns that must be copied exactly.
    extracted_path = resolve_existing_path(
        ["./data/extracted_outputs/df3.csv", "./data/extracted_inputs/df3.csv"],
        "doc3 extracted source",
    )

    output_path = Path("./data/generated_outputs/doc3.csv")
    pre_df = pd.read_csv(preprocessed_path)
    ext_df = pd.read_csv(extracted_path)

    # Match rows primarily by Number.
    ext_lookup = build_row_lookup(ext_df, ["Number", "Risk ID", "Reference"])

    rows = []
    for i, pre_row in pre_df.iterrows():
        # Build LLM context from preprocessed content.
        number = get_value(pre_row, "Number", "")
        llm_desc = get_value(pre_row, "Risk Description", "")

        # Resolve extracted row for copied columns.
        ext_row = ext_lookup.get(normalize_key(number))
        if ext_row is None and i < len(ext_df):
            ext_row = ext_df.iloc[i]

        # Keep Number and Risk Description exactly from extracted input.
        out_number = get_value(ext_row, "Number", number)
        out_desc = get_value(ext_row, "Risk Description", llm_desc)

        rows.append({
            "Date Added":                    "",
            "Number":                        out_number,
            "Risk Description":              out_desc,
            "Project Stage":                 infer_project_stage_doc3(out_number, llm_desc),
            "Project Category":              infer_project_category_doc3(out_number, llm_desc),
            "Risk Owner":                    infer_risk_owner_doc3(out_number, llm_desc),
            "Likelihood (1-10)":             get_value(ext_row, "Probability", ""),
            "Impact (1-10)":                 get_value(ext_row, "Severity", ""),
            "Risk Priority (low, med, high)": get_value(ext_row, "Score", ""),
            "Mitigating Action":             get_value(ext_row, "Action Plan", ""),
        })
        print(f"  doc3 processed: {out_number}")

    # Apply title-case formatting only to selected metadata columns.
    output_df = apply_title_case_to_selected_columns(pd.DataFrame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Doc3 done → {output_path}")


# Generate doc4 output with mixed copied and inferred fields.
def generate_doc4():
    # Use preprocessed data as the source for LLM understanding.
    preprocessed_path = resolve_existing_path(
        ["./data/preprocessed_outputs/df4.csv", "./data/preprocessed_inputs/df4.csv"],
        "doc4 preprocessed source",
    )

    # Use extracted data for fields that should be copied directly.
    extracted_path = resolve_existing_path(
        ["./data/extracted_outputs/df4.csv", "./data/extracted_inputs/df4.csv"],
        "doc4 extracted source",
    )

    output_path = Path("./data/generated_outputs/doc4.csv")
    pre_df = pd.read_csv(preprocessed_path)
    ext_df = pd.read_csv(extracted_path)

    # Create a lookup to align copied columns even if row order differs.
    ext_lookup = build_row_lookup(ext_df, ["Risk ID", "Reference"])

    rows = []
    for i, pre_row in pre_df.iterrows():
        # Build LLM inputs from preprocessed content.
        risk_id = get_value(pre_row, "Risk ID", get_value(pre_row, "Reference", ""))
        desc = get_value(pre_row, "Risk Description", get_value(pre_row, "Risk", ""))

        # Resolve extracted row for copied fields.
        ext_row = ext_lookup.get(normalize_key(risk_id))
        if ext_row is None and i < len(ext_df):
            ext_row = ext_df.iloc[i]

        likelihood = LIKELIHOOD_MAP.get(str(get_value(ext_row, "Likelihood (1-10)", "")).strip().lower(), "")
        impact = IMPACT_MAP.get(str(get_value(ext_row, "Impact (1-10)", "")).strip().lower(), "")

        rows.append({
            "Risk ID":                        risk_id,
            "Risk Description":               desc,
            "Project Stage":                  infer_project_stage_doc4(risk_id, desc),
            "Project Category":               infer_project_category_doc4(risk_id, desc),
            "Risk Owner":                     infer_risk_owner_doc4(risk_id, desc),
            "Mitigating Action":              get_value(ext_row, "Mitigating Action", ""),
            "Likelihood (1-10)":              likelihood,
            "Impact (1-10)":                  impact,
            "Risk Priority (low, med, high)": str(get_value(ext_row, "Risk Priority (low, med, high)", "")).strip().lower(),
        })
        print(f"  doc4 processed: {risk_id}")

    # Apply title-case formatting only to selected metadata columns.
    output_df = apply_title_case_to_selected_columns(pd.DataFrame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Doc4 done → {output_path}")


# Generate doc5 output with 1-5 matrix priorities.
def generate_doc5():
    # Use preprocessed data as the source for LLM understanding.
    preprocessed_path = resolve_existing_path(
        ["./data/preprocessed_outputs/df5.csv", "./data/preprocessed_inputs/df5.csv"],
        "doc5 preprocessed source",
    )

    # Use extracted data for fields that should be copied directly.
    extracted_path = resolve_existing_path(
        ["./data/extracted_outputs/df5.csv", "./data/extracted_inputs/df5.csv"],
        "doc5 extracted source",
    )

    output_path = Path("./data/generated_outputs/doc5.csv")
    pre_df = pd.read_csv(preprocessed_path)
    ext_df = pd.read_csv(extracted_path)

    # Create a lookup to align copied columns even if row order differs.
    ext_lookup = build_row_lookup(ext_df, ["Reference", "Risk ID"])

    # Clean description text from the LLM source.
    desc_series = (
        pre_df.get("Risk", pre_df.get("Risk Description", "")).fillna("").astype(str).str.strip()
        .str.replace(r"^[^a-zA-Z]+", "", regex=True)
    )

    rows = []
    for i, pre_row in pre_df.iterrows():
        # Build LLM inputs from preprocessed content.
        risk_id = get_value(pre_row, "Reference", get_value(pre_row, "Risk ID", ""))
        desc      = desc_series[i]
        raw_owner = str(get_value(pre_row, "Risk_Owner", ""))

        # Resolve extracted row for copied fields.
        ext_row = ext_lookup.get(normalize_key(risk_id))
        if ext_row is None and i < len(ext_df):
            ext_row = ext_df.iloc[i]

        rows.append({
            "Risk ID":                        risk_id,
            "Risk Description":               desc,
            "Project Stage":                  infer_project_stage_doc5(risk_id, desc),
            "Project Category":               infer_project_category_doc5(risk_id, desc),
            "Risk Owner":                     infer_risk_owner_doc5(risk_id, desc, raw_owner),
            "Likelihood No Action (1-5)":     get_value(ext_row, "Risk_if_No_Action_Likelihood", ""),
            "Impact No Action (1-5)":         get_value(ext_row, "Risk_if_No_Action_Impact", ""),
            "Risk Priority No Action (low, med, high)": calc_priority_doc5_scale_1_5(
                get_value(ext_row, "Risk_if_No_Action_Likelihood", ""),
                get_value(ext_row, "Risk_if_No_Action_Impact", ""),
            ),
            "Mitigating Action":              get_value(ext_row, "Actions_Being_Taken", ""),
            "Likelihood Current (1-5)":       get_value(ext_row, "Current_Risk_Likelihood", ""),
            "Impact Current (1-5)":           get_value(ext_row, "Current_Risk_Impact", ""),
            # Doc5 uses a 1-5 scale for likelihood and impact.
            "Risk Priority Current (low, med, high)": calc_priority_doc5_scale_1_5(
                get_value(ext_row, "Current_Risk_Likelihood", ""),
                get_value(ext_row, "Current_Risk_Impact", ""),
            ),
        })
        print(f"  doc5 processed: {risk_id}")

    # Apply title-case formatting only to selected metadata columns.
    output_df = apply_title_case_to_selected_columns(pd.DataFrame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Doc5 done → {output_path}")


# Run all document generators in sequence.
def run_generating_outputs():
    # Run both generators in sequence to produce final csv outputs.
    generate_doc1()
    generate_doc2()
    generate_doc3()
    generate_doc4()
    generate_doc5()

if __name__ == "__main__":
    run_generating_outputs()