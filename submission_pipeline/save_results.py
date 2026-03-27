"""
submission_pipeline/save_results.py
-----------------------------------
Excel Export & Formatting Module.
Converts AI JSON list into final .xlsx files 
with proper column order for Judges.
"""

import pandas as pd

def format_and_save_final_excel(json_data_list, output_filepath):
    """
    Convert LLM inference JSON list to Excel format.
    """
    if not json_data_list:
        print("⚠️ Warning: Empty JSON data, no Excel saved.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(json_data_list)

    # 🎯 STRICT TARGET COLUMNS FOR JUDGES
    # Combines Document 1-4 and Document 5 format (Deira's Logic)
    target_columns = [
        "Risk ID",
        "Risk Description",
        "Project Stage",
        "Project Category",
        "Risk Owner",
        "Mitigating Action",
        
        # 1-10 Scale Columns (Documents 1, 2, 3, 4)
        "Likelihood (1-10) (pre-mitigation)",
        "Impact (1-10) (pre-mitigation)",
        "Risk Priority (pre-mitigation)",
        "Likelihood (1-10) (post-mitigation)",
        "Impact (1-10) (post-mitigation)",
        "Risk Priority (post-mitigation)",
        
        # 1-5 Scale Columns (SPECIFIC to Document 5 - Deira's Logic)
        "Likelihood No Action (1-5)",
        "Impact No Action (1-5)",
        "Risk Priority No Action (low, med, high)",
        "Likelihood Current (1-5)",
        "Impact Current (1-5)",
        "Risk Priority Current (low, med, high)",

        # AI Log (Optional but good for Audit)
        "Schema Alignment Log",
        "Risk ID (Reasoning)",
        "Risk Description (Reasoning)",
        "Project Stage (Reasoning)",
        "Project Category (Reasoning)",
        "Risk Owner (Reasoning)",
        "Likelihood (Reasoning)",
        "Impact (Reasoning)",
        "Mitigating Action (Reasoning)"
    ]

    # Filter only available columns to ensure order
    available_cols = [col for col in target_columns if col in df.columns]
    
    # Append leftover columns to the far right (if any)
    extra_cols = [col for col in df.columns if col not in available_cols]
    final_cols = available_cols + extra_cols

    df_final = df[final_cols]

    # Save to Excel
    try:
        df_final.to_excel(output_filepath, index=False)
    except Exception as e:
        print(f"⚠️ Failed to save Excel: {e}")