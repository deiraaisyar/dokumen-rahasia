"""
submission_pipeline/save_results.py
-----------------------------------
Final Formatting Module.
Ensures the output Excel perfectly matches the Judge's expectations 
without any debugging or reasoning columns cluttering the view.
"""

import pandas as pd

def format_and_save_final_excel(json_data_list, output_filepath):
    if not json_data_list:
        print("⚠️ Warning: Data JSON kosong, tidak ada Excel yang disimpan.")
        return

    df = pd.DataFrame(json_data_list)

    # 🎯 TARGET KOLOM MUTLAK UNTUK JURI (Tanpa Kolom Reasoning!)
    target_columns = [
        "Risk ID",
        "Risk Description",
        "Project Stage",
        "Project Category",
        "Risk Owner",
        "Mitigating Action",
        
        # Kolom Skala 1-10 (Dokumen 1, 2, 3, 4)
        "Likelihood (1-10) (pre-mitigation)",
        "Impact (1-10) (pre-mitigation)",
        "Risk Priority (pre-mitigation)",
        "Likelihood (1-10) (post-mitigation)",
        "Impact (1-10) (post-mitigation)",
        "Risk Priority (post-mitigation)",
        
        # Kolom Skala 1-5 (KHUSUS Dokumen 5)
        "Likelihood No Action (1-5)",
        "Impact No Action (1-5)",
        "Risk Priority No Action (low, med, high)",
        "Likelihood Current (1-5)",
        "Impact Current (1-5)",
        "Risk Priority Current (low, med, high)"
    ]

    # Filter hanya kolom yang ada di list target (Otomatis membuang kolom Reasoning dari AI)
    available_cols = [col for col in target_columns if col in df.columns]
    df_final = df[available_cols]

    try:
        df_final.to_excel(output_filepath, index=False)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan Excel: {e}")