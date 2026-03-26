import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_risk_owner(df):
    if "Risk Owner" in df.columns:
        names = []
        roles = []

        for val in df["Risk Owner"]:
            if pd.isna(val):
                names.append(val)
                roles.append(val)
                continue

            val = str(val)
            match = re.search(r"\((.*?)\)", val)

            if match:
                role = match.group(1)
                name = re.sub(r"\(.*?\)", "", val).strip()
            else:
                role = val
                name = val

            names.append(name)
            roles.append(role)

        df["Risk Owner Name"] = names
        df["Risk Owner"] = roles

    return df

def preprocess_dataframe(df):
    df = df.dropna(axis=1, how="all")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_text)

    return df

def run_preprocessing_outputs():
    input_dir = Path("./data/extracted_outputs")
    output_dir = Path("./data/preprocessed_outputs")
    output_dir.mkdir(exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print("no extracted output files found")
        return

    for file in csv_files:
        print(f"processing output {file.name}")
        df = pd.read_csv(file)

        if "IVC" in file.name:
            df = df.iloc[1:].reset_index(drop=True)

        df = preprocess_dataframe(df)

        output_path = output_dir / file.name
        df.to_csv(output_path, index=False)

        print(f"saved {output_path}")
        
def run_preprocessing():
    input_dir = Path("./data/extracted_inputs")
    output_dir = Path("./data/preprocessed_inputs")
    output_dir.mkdir(exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print("no extracted files found")
        return

    for file in csv_files:
        print(f"processing {file.name}")
        df = pd.read_csv(file)

        if file.name == "df1.csv":
            df = split_risk_owner(df)
            if "Baseline +/-" in df.columns:
                df = df.drop(columns=["Baseline +/-"])

        df = preprocess_dataframe(df)

        output_path = output_dir / file.name
        df.to_csv(output_path, index=False)

        print(f"saved {output_path}")
        run_preprocessing_outputs()

if __name__ == "__main__":
    run_preprocessing()