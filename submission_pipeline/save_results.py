"""
submission_pipeline/save_results.py
-----------------------------------
Modul Ekspor & Formatting Excel.
Berfungsi mengubah list JSON dari AI menjadi file .xlsx final 
dengan urutan kolom yang rapi untuk Juri.
"""

import pandas as pd

def format_and_save_final_excel(json_data_list, output_filepath):
    """
    Mengubah list JSON hasil inferensi LLM menjadi format Excel.
    """
    if not json_data_list:
        print("⚠️ Warning: Data JSON kosong, tidak ada Excel yang disimpan.")
        return

    # Jadikan DataFrame
    df = pd.DataFrame(json_data_list)

    # 🎯 TARGET KOLOM MUTLAK UNTUK JURI
    # Menggabungkan format Dokumen 1-4 dan Dokumen 5 (Logika Kak Deira)
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
        
        # Kolom Skala 1-5 (KHUSUS Dokumen 5 - Logika Deira)
        "Likelihood No Action (1-5)",
        "Impact No Action (1-5)",
        "Risk Priority No Action (low, med, high)",
        "Likelihood Current (1-5)",
        "Impact Current (1-5)",
        "Risk Priority Current (low, med, high)",

        # Log AI (Opsional tapi bagus untuk Audit)
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

    # Filter hanya kolom yang ada di list target agar urutannya rapi
    available_cols = [col for col in target_columns if col in df.columns]
    
    # Masukkan kolom sisa yang mungkin terlewat (jika ada) di bagian paling kanan
    extra_cols = [col for col in df.columns if col not in available_cols]
    final_cols = available_cols + extra_cols

    df_final = df[final_cols]

    # Simpan ke Excel
    try:
        df_final.to_excel(output_filepath, index=False)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan Excel: {e}")