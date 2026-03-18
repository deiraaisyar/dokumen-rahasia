import pandas as pd
from pathlib import Path

def generate_doc4():
    input_path = Path("./data/extracted_inputs/df4.csv")
    output_path = Path("./data/generated_outputs/doc4.csv")
    df = pd.read_csv(input_path)

    out = pd.DataFrame({
        "Risk ID": df["Risk ID"] if "Risk ID" in df.columns else df.iloc[:,0],
        "Risk Description": df["Risk Description"] if "Risk Description" in df.columns else df.iloc[:,1],
        "Project Stage": "",
        "Project Category": "",
        "Risk Owner": "",
        "Mitigating Action": "",
        "Likelihood (1-10)": "",
        "Impact (1-10)": "",
        "Risk Priority (low, med, high)": df["Risk Priority"] if "Risk Priority" in df.columns else ""
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return out

def generate_doc5():
    input_path = Path("./data/extracted_inputs/df5.csv")
    output_path = Path("./data/generated_outputs/doc5.csv")

    df = pd.read_csv(input_path)

    n = len(df) // 2
    df_no = df.iloc[:n].reset_index(drop=True)
    df_cur = df.iloc[n:].reset_index(drop=True)

    desc = df_no["Risk"].fillna("").astype(str).str.strip()
    desc = desc.str.replace(r'^[^a-zA-Z]+', '', regex=True)

    out = pd.DataFrame({
        "Risk ID": df_no["Reference"],
        "Risk Description": desc,
        "Project Stage": "",
        "Project Category": "",
        "Risk Owner": "",
        "Mitigating Action No Action": df_no["Mitigation"],
        "Mitigating Action Current": df_cur["Actions_Being_Taken"],
        "Likelihood No Action (1-10)": df_no["Risk_if_No_Action_Likelihood"],
        "Impact No Action (1-10)": df_no["Risk_if_No_Action_Impact"],
        "Likelihood Current (1-10)": df_cur["Current_Risk_Likelihood"],
        "Impact Current (1-10)": df_cur["Current_Risk_Impact"],
        "Risk Priority (low, med, high)": ""
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return out

def run_generating_outputs():
    generate_doc4()
    generate_doc5()