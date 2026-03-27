"""
submission_pipeline/preprocessing.py
------------------------------------
Data cleaning and normalization module.
Responsible for standardizing text, handling missing values, and restructuring specific columns 
so that downstream evaluation and processing have uniform data structures.
"""

import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    # Return immediately if the value is NaN/null to avoid type errors
    if pd.isna(text):
        return text
    # Convert value to string to ensure string operations work
    text = str(text)
    # Convert all characters to lowercase to maintain uniformity
    text = text.lower()
    # Replace any non-alphanumeric character (punctuation, symbols) with a space
    text = re.sub(r"[^\w\s]", " ", text)
    # Replace multiple consecutive spaces with a single space and trim edges
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_risk_owner(df):
    # Check if the 'Risk Owner' column exists in the dataframe before operating
    if "Risk Owner" in df.columns:
        names = []
        roles = []

        # Iterate through each value in the Risk Owner column
        for val in df["Risk Owner"]:
            # If the value is missing, append it as is and skip to the next iteration
            if pd.isna(val):
                names.append(val)
                roles.append(val)
                continue

            # Cast value to string for regex parsing
            val = str(val)
            # Look for text inside parentheses, e.g., "John Doe (Project Manager)"
            match = re.search(r"\((.*?)\)", val)

            # If parentheses are found, extract the role and remove it from the name
            if match:
                role = match.group(1) # The text inside the parentheses
                name = re.sub(r"\(.*?\)", "", val).strip() # The text outside
            # If no parentheses are found, use the whole string for both name and role as a fallback
            else:
                role = val
                name = val

            names.append(name)
            roles.append(role)

        # Assign the separated lists back to the dataframe
        df["Risk Owner Name"] = names
        df["Risk Owner"] = roles # Overwrite with the extracted role

    return df

def preprocess_dataframe(df):
    # Drop columns where ALL values are completely missing (NaN)
    df = df.dropna(axis=1, how="all")

    # Select only columns with object (string) data types and apply the clean_text function column-wise
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_text)

    return df

def run_preprocessing_outputs():
    # Define paths for raw outputs and their designated processed directory
    input_dir = Path("./extracted_outputs")
    output_dir = Path("./preprocessed_outputs")
    # Ensure the preprocessed outputs directory exists
    output_dir.mkdir(exist_ok=True)

    # Retrieve all CSV files in the extracted outputs folder
    csv_files = list(input_dir.glob("*.csv"))
    
    # Guard clause: if no files are found, log string and exit function
    if not csv_files:
        print("no extracted output files found")
        return

    # Iterate over each CSV file found
    for file in csv_files:
        print(f"processing output {file.name}")
        # Load the CSV into a pandas dataframe
        df = pd.read_csv(file)

        # Specific rule for Document 1 (IVC): drop the first row (usually a sub-header) and reset index
        if "IVC" in file.name:
            df = df.iloc[1:].reset_index(drop=True)

        # Apply general column cleaning and NaN dropping
        df = preprocess_dataframe(df)

        # Construct the save path and export the cleaned data back to CSV
        output_path = output_dir / file.name
        df.to_csv(output_path, index=False)

        print(f"saved {output_path}")
        
def run_preprocessing():
    # Define paths for raw inputs and their designated processed directory
    input_dir = Path("./extracted_inputs")
    output_dir = Path("./preprocessed_inputs")
    # Ensure the preprocessed inputs directory exists
    output_dir.mkdir(exist_ok=True)

    # Retrieve all CSV files in the extracted inputs folder
    csv_files = list(input_dir.glob("*.csv"))
    
    # Guard clause: if no files are found, log string and exit function
    if not csv_files:
        print("no extracted files found")
        return

    # Iterate over each CSV file found
    for file in csv_files:
        print(f"processing {file.name}")
        # Load the CSV into a pandas dataframe
        df = pd.read_csv(file)

        # Specific rules for df1.csv (the primary IVC document)
        if file.name == "df1.csv":
            # Apply the owner/role splitting logic
            df = split_risk_owner(df)
            # Drop the "Baseline +/-" column if it exists as it's not needed for downstream evaluations
            if "Baseline +/-" in df.columns:
                df = df.drop(columns=["Baseline +/-"])

        # Apply general text cleaning and strip out empty columns
        df = preprocess_dataframe(df)

        # Construct the save path and export the cleaned data back to CSV
        output_path = output_dir / file.name
        df.to_csv(output_path, index=False)

        print(f"saved {output_path}")

    # Trigger the output preprocessing pipeline sequentially after inputs are done
    run_preprocessing_outputs()

# If the script is run directly, execute the main inputs pipeline
if __name__ == "__main__":
    run_preprocessing()