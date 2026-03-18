import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def load_valid_values() -> tuple:
    """Load valid Project Stage, Category, Risk Owner from existing output files (doc1-3)."""
    data_dir = Path("./data/outputs")
    stage_vals, category_vals, owner_vals = [], [], []

    for file in sorted(data_dir.glob("*.xlsx")):
        df = pd.read_excel(file)
        if "Project Stage" in df.columns:
            stage_vals.extend(df["Project Stage"].dropna().astype(str).unique().tolist())
        if "Project Category" in df.columns:
            category_vals.extend(df["Project Category"].dropna().astype(str).unique().tolist())
        if "Risk Owner" in df.columns:
            owner_vals.extend(df["Risk Owner"].dropna().astype(str).unique().tolist())

    return list(set(stage_vals)), list(set(category_vals)), list(set(owner_vals))


VALID_STAGES, VALID_CATEGORIES, VALID_OWNERS = load_valid_values()

LIKELIHOOD_MAP = {"rare": 2, "unlikely": 4, "possible": 6, "likely": 8, "almost certain": 10}
IMPACT_MAP     = {"minor": 2, "serious": 5, "major": 8, "critical": 10}

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

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str) -> dict:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


def calc_priority(likelihood, impact) -> str:
    try:
        score = float(likelihood) * float(impact)
        if score <= 20:   return "Low"
        elif score <= 50: return "Med"
        return "High"
    except Exception:
        return ""


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

def generate_doc4():
    input_path  = Path("./data/extracted_inputs/df4.csv")
    output_path = Path("./data/generated_outputs/doc4.csv")
    df = pd.read_csv(input_path)

    rows = []
    for _, row in df.iterrows():
        risk_id = row.get("Risk ID", "")
        desc    = row.get("Risk Description", "")

        likelihood = LIKELIHOOD_MAP.get(str(row.get("Likelihood (1-10)", "")).strip().lower(), "")
        impact     = IMPACT_MAP.get(str(row.get("Impact (1-10)", "")).strip().lower(), "")

        rows.append({
            "Risk ID":                        risk_id,
            "Risk Description":               desc,
            "Project Stage":                  infer_project_stage_doc4(risk_id, desc),
            "Project Category":               infer_project_category_doc4(risk_id, desc),
            "Risk Owner":                     infer_risk_owner_doc4(risk_id, desc),
            "Mitigating Action":              row.get("Mitigating Action", ""),
            "Likelihood (1-10)":              likelihood,
            "Impact (1-10)":                  impact,
            "Risk Priority (low, med, high)": str(row.get("Risk Priority (low, med, high)", "")).strip().lower(),
        })
        print(f"  doc4 processed: {risk_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Doc4 done → {output_path}")


def generate_doc5():
    input_path  = Path("./data/extracted_inputs/df5.csv")
    output_path = Path("./data/generated_outputs/doc5.csv")
    df = pd.read_csv(input_path)

    # each row already has both pre and post mitigation columns — no split needed
    desc_series = (
        df["Risk"].fillna("").astype(str).str.strip()
        .str.replace(r"^[^a-zA-Z]+", "", regex=True)
    )

    rows = []
    for i, row in df.iterrows():
        risk_id   = row["Reference"]
        desc      = desc_series[i]
        raw_owner = str(row["Risk_Owner"]) if "Risk_Owner" in df.columns else ""

        rows.append({
            "Risk ID":                        risk_id,
            "Risk Description":               desc,
            "Project Stage":                  infer_project_stage_doc5(risk_id, desc),
            "Project Category":               infer_project_category_doc5(risk_id, desc),
            "Risk Owner":                     infer_risk_owner_doc5(risk_id, desc, raw_owner),
            "Likelihood No Action (1-10)":    row["Risk_if_No_Action_Likelihood"],
            "Impact No Action (1-10)":        row["Risk_if_No_Action_Impact"],
            "Mitigating Action":              row["Actions_Being_Taken"],
            "Likelihood Current (1-10)":      row["Current_Risk_Likelihood"],
            "Impact Current (1-10)":          row["Current_Risk_Impact"],
            "Risk Priority (low, med, high)": calc_priority(row["Current_Risk_Likelihood"], row["Current_Risk_Impact"]),
        })
        print(f"  doc5 processed: {risk_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Doc5 done → {output_path}")


def run_generating_outputs():
    generate_doc4()
    generate_doc5()


if __name__ == "__main__":
    run_generating_outputs()