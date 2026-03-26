"""
src/token_tracking.py
---------------------
Modul untuk pelacakan penggunaan token dan estimasi biaya API secara Thread-Safe.
Fokus Utama:
1. Menghitung jumlah token dari teks secara offline (menggunakan tiktoken).
2. Melacak total penggunaan token prompt dan completion selama eksekusi paralel.
3. Menghitung estimasi biaya berdasarkan harga model LLM.
"""

import threading
import warnings

try:
    import tiktoken
except ImportError:
    tiktoken = None
    warnings.warn("⚠️ Library 'tiktoken' belum terinstall. Estimasi token offline akan menggunakan pendekatan jumlah kata (kurang presisi). Jalankan: pip install tiktoken")

# ==============================================================================
# 1. KONFIGURASI HARGA (DEEPSEEK V3 / GROQ COMPATIBLE)
# Harga per 1 juta token (dalam USD)
# ==============================================================================
PRICING = {
    "deepseek-chat": {
        "prompt": 0.14,
        "completion": 0.28
    },
    "llama-3.1-8b-instant": {
        "prompt": 0.05,
        "completion": 0.08
    }
}


# ==============================================================================
# 2. THREAD-SAFE TOKEN TRACKER (SINGLETON)
# ==============================================================================
class TokenTracker:
    """
    Kelas Thread-Safe untuk mencatat penggunaan token
    dari seluruh proses prediksi paralel (multithreading).
    """
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self._lock = threading.Lock()
        
        self.tokenizer = None
        if tiktoken:
            try:
                # cl100k_base adalah standar encoding GPT-4, sangat mendekati arsitektur LLM modern lainnya
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"⚠️ Gagal memuat tokenizer: {e}")

    def count_tokens(self, text: str) -> int:
        """Menghitung estimasi jumlah token dari sebuah teks secara offline."""
        if not text:
            return 0
            
        text_str = str(text)
        if self.tokenizer:
            return len(self.tokenizer.encode(text_str))
        else:
            # Fallback kasar: 1 token ~ 0.75 kata
            return int(len(text_str.split()) / 0.75)

    def truncate_text(self, text: str, max_tokens: int = 2000) -> str:
        """
        Penyelamat API (Safeguard):
        Memotong teks jika melebihi batas token yang diizinkan.
        Berguna untuk mencegah error API (Context Length Exceeded).
        """
        if self.count_tokens(text) <= max_tokens:
            return text
            
        print(f"✂️ Peringatan: Teks melebihi {max_tokens} token. Melakukan Truncation otomatis...")
        text_str = str(text)
        
        if self.tokenizer:
            encoded = self.tokenizer.encode(text_str)
            truncated = self.tokenizer.decode(encoded[:max_tokens])
            return truncated + "\n... [TRUNCATED DUE TO LENGTH]"
        else:
            # Fallback pemotongan berbasis karakter (1 token ≈ 4 karakter)
            max_chars = max_tokens * 4
            return text_str[:max_chars] + "\n... [TRUNCATED]"

    def log_usage(self, prompt_tokens: int, completion_tokens: int):
        """Mencatat penggunaan token dari response API secara aman antar thread."""
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1

    def get_estimated_cost(self, model="deepseek-chat") -> float:
        """Menghitung estimasi total biaya dalam USD."""
        with self._lock:
            rates = PRICING.get(model, PRICING["deepseek-chat"])
            prompt_cost = (self.total_prompt_tokens / 1_000_000) * rates["prompt"]
            completion_cost = (self.total_completion_tokens / 1_000_000) * rates["completion"]
            return prompt_cost + completion_cost

    def print_summary(self, model="deepseek-chat"):
        """Mencetak ringkasan penggunaan token dan biaya ke terminal."""
        cost = self.get_estimated_cost(model)
        
        with self._lock:
            prompt_t = self.total_prompt_tokens
            comp_t = self.total_completion_tokens
            calls = self.total_calls

        print("\n" + "="*50)
        print("💰 LAPORAN PENGGUNAAN API & TOKEN (LLM)")
        print("="*50)
        print(f"Model Digunakan       : {model}")
        print(f"Total API Calls       : {calls:,} requests")
        print(f"Total Prompt Tokens   : {prompt_t:,} tokens")
        print(f"Total Output Tokens   : {comp_t:,} tokens")
        print(f"Total Keseluruhan     : {prompt_t + comp_t:,} tokens")
        print("-" * 50)
        print(f"Estimasi Total Biaya  : ${cost:.6f} USD")
        if calls > 0:
            print(f"Rata-rata Biaya/Call  : ${(cost / calls):.6f} USD")
        print("="*50 + "\n")


# ==============================================================================
# 3. GLOBAL WRAPPER FUNCTIONS (Untuk diimpor oleh modul lain)
# ==============================================================================
# Inisialisasi instance global agar state-nya tetap sama di seluruh proyek
tracker = TokenTracker()

def count_tokens(text: str) -> int:
    return tracker.count_tokens(text)

def truncate_text(text: str, max_tokens: int = 2000) -> str:
    return tracker.truncate_text(text, max_tokens)

def log_api_usage(prompt_tokens: int, completion_tokens: int):
    tracker.log_usage(prompt_tokens, completion_tokens)

def print_token_summary(model="deepseek-chat"):
    tracker.print_summary(model)


# ==============================================================================
# DEBUG / TESTING LOKAL
# ==============================================================================
if __name__ == "__main__":
    print("--- 🛠️ Menguji Modul Token Tracking ---")
    
    # 1. Tes Menghitung Token
    sample_text = "Proyek ini mengalami risiko pembengkakan anggaran karena kegagalan pengiriman vendor."
    tokens = count_tokens(sample_text)
    print(f"\nTeks: '{sample_text}'")
    print(f"Jumlah Token Estimasi (Offline): {tokens} tokens")
    
    # 2. Tes Truncate Teks
    long_text = "Risk Description: " + ("Error data " * 500)
    print(f"\nPanjang Teks Asli: {count_tokens(long_text)} tokens")
    safe_text = truncate_text(long_text, max_tokens=20)
    print(f"Hasil Truncate (max 20): {safe_text}")
    
    # 3. Tes Thread-Safe Logging
    print("\n[Simulasi] Menerima balasan dari API sebanyak 3 kali (Multithreading)...")
    log_api_usage(prompt_tokens=1500, completion_tokens=50)
    log_api_usage(prompt_tokens=1450, completion_tokens=45)
    log_api_usage(prompt_tokens=1600, completion_tokens=60)
    
    # 4. Tampilkan Laporan Biaya
    print_token_summary()