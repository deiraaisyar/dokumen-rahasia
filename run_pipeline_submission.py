"""
run_pipeline_submission.py
Master Orchestrator - OECD NEA Coding Competition
[THE ULTIMATE MERGE: Deira's Logic + Hybrid AI Engine]
"""
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 1. Pastikan folder submission_pipeline/ terbaca oleh Python
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "submission_pipeline")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# 2. Import modul-modul kita yang sudah digabung
from extracting import extract_excel, extract_pdf, preprocessing, save_extracted, format_df_to_llm_text
from generating_outputs import process_single_risk, save_cache_to_disk
from save_results import format_and_save_final_excel
from evaluating import run_evaluating

def main():
    print("="*70)
    print("🚀 FASE 1: EKSTRAKSI & PREPROCESSING (DEIRA + AI VISION)")
    print("="*70)
    
    try:
        # Menjalankan mesin penarik data gabungan
        df1, df2, df3, df4 = extract_excel()
        df5 = extract_pdf()
        
        # Menjalankan pembersih data
        df1 = preprocessing(df1, "df1")
        df2 = preprocessing(df2, "df2")
        df3 = preprocessing(df3, "df3")
        df4 = preprocessing(df4, "df4")
        df5 = preprocessing(df5, "df5")
        
        # Menyimpan CSV Audit (SOP Deira)
        save_extracted([df1, df2, df3, df4, df5])
        print("✅ Ekstraksi & Preprocessing Sukses!")
    except Exception as e:
        print(f"❌ Gagal di Fase Ekstraksi: {e}")
        return

    print("\n" + "="*70)
    print("🧠 FASE 2: PREDIKSI AI (HYBRID TRANSFORMER)")
    print("="*70)
    
    OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Pemetaan Dataframe dengan Nama Output Excel Juri
    dokumen_map = {
        "1. IVC DOE R2": df1,
        "2. City of York Council": df2,
        "3. Digital Security IT Sample Register": df3,
        "4. Moorgate Crossrail Register": df4,
        "5. Corporate Risk Register": df5
    }
    
    for doc_name, df_clean in dokumen_map.items():
        if df_clean is None or df_clean.empty:
            print(f"⚠️ Skip {doc_name}: Data kosong.")
            continue
            
        print(f"\n🚀 Memproses AI untuk: {doc_name}")
        
        # Ubah tabel jadi teks Prompt LLM
        llm_texts = format_df_to_llm_text(df_clean)
        
        final_results = []
        for text in tqdm(llm_texts, desc=f"Inferensi AI {doc_name[:15]}"):
            # Proses Prediksi (Mengirim teks ke AI beserta konteks nama dokumen)
            res = process_single_risk(target_text=text, project_name=doc_name)
            final_results.append(res)
            
        final_excel_name = f"{doc_name} (Final).xlsx"
        final_excel_path = OUTPUT_DIR / final_excel_name
        
        try:
            # Menyaring dan format Excel pakai save_results.py buatanmu
            format_and_save_final_excel(final_results, str(final_excel_path))
            print(f"✅ Berhasil menyimpan: {final_excel_name}")
        except Exception as e:
            print(f"⚠️ Gagal strict formatting, memakai fallback Pandas: {e}")
            pd.DataFrame(final_results).to_excel(str(final_excel_path), index=False)
            print(f"✅ Berhasil menyimpan (Fallback): {final_excel_name}")
            
    # Menyimpan memori AI ke file JSON
    try: save_cache_to_disk()
    except: pass
    
    print("\n" + "="*70)
    print("📊 FASE 3: EVALUASI PERFORMA AI (BENCHMARKING)")
    print("="*70)
    try:
        run_evaluating()
    except Exception as e:
        print(f"⚠️ Laporan Evaluasi diskip karena error minor: {e}")
        print("💡 Jangan panik, Excel Final untuk Juri tetap aman di folder outputs!")
        
    print("\n🏆 SELURUH PIPELINE SELESAI! SIAP DIKUMPULKAN KE PANITIA! 🏆")

if __name__ == "__main__":
    main()