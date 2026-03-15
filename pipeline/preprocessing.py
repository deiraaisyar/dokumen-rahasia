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

def preprocess_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_text)
    return df

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
        df = preprocess_dataframe(df)
        output_path = output_dir / file.name
        df.to_csv(output_path, index=False)
        print(f"saved {output_path}")


if __name__ == "__main__":
    run_preprocessing()