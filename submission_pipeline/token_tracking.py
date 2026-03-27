"""
submission_pipeline/token_tracking.py
-------------------------------------
Thread-Safe Token Tracking and API Cost Estimation Module.
Core Focus:
1. Offline token counting (via tiktoken).
2. Tracking cumulative prompt and completion tokens during parallel multi-threading execution.
3. Estimating API monetary cost based on standard model pricing.
"""

import threading
import warnings

try:
    import tiktoken
except ImportError:
    tiktoken = None
    # Warn user if tokenizer is missing; fallback to rough word-count approach instead of crashing
    warnings.warn("⚠️ 'tiktoken' library is not installed. Offline token estimation will use word count approach (less precise). Run: pip install tiktoken")

# ==============================================================================
# 1. PRICING CONFIGURATION (DEEPSEEK V3 / GROQ COMPATIBLE)
# Price per 1 Million Tokens (in USD)
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
    Thread-Safe class to centrally record token usage
    accrued from asynchronous parallel API calls.
    """
    def __init__(self):
        # Initialize cumulative usage variables
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        # Initialize thread lock to prevent data race conditions when multiple workers log usage
        self._lock = threading.Lock()
        
        self.tokenizer = None
        # Attempt to load the OpenAI standard tokenizer if available
        if tiktoken:
            try:
                # cl100k_base represents GPT-4 encoding; highly analogous to modern generic LLM tokenizations
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"⚠️ Failed to load tokenizer: {e}")

    def count_tokens(self, text: str) -> int:
        """Calculates an offline estimation of how many tokens a string consumes."""
        # Return 0 safely if text is null/empty
        if not text:
            return 0
            
        # Cast input to string as safety precaution
        text_str = str(text)
        # Use precise tokenizer if loaded successfully
        if self.tokenizer:
            return len(self.tokenizer.encode(text_str))
        else:
            # Rough math fallback: assume 1 token represents roughly 0.75 words on average
            return int(len(text_str.split()) / 0.75)

    def truncate_text(self, text: str, max_tokens: int = 2000) -> str:
        """
        API Safeguard System:
        Forcefully chunks text if it exceeds arbitrary token limitations.
        Crucial for preventing sudden crash errors like 'Context Length Exceeded'.
        """
        # If the payload fits comfortably within bounds, return it unmodified
        if self.count_tokens(text) <= max_tokens:
            return text
            
        # Warn system that truncation is dynamically occurring
        print(f"✂️ Warning: Text exceeds {max_tokens} tokens. Executing automatic truncation...")
        text_str = str(text)
        
        if self.tokenizer:
            # Sub-slice via tokenizer encoding matrix for high precision truncation
            encoded = self.tokenizer.encode(text_str)
            truncated = self.tokenizer.decode(encoded[:max_tokens])
            return truncated + "\n... [TRUNCATED DUE TO LENGTH]"
        else:
            # Fallback mathematical truncation (estimating roughly 4 characters per token length)
            max_chars = max_tokens * 4
            return text_str[:max_chars] + "\n... [TRUNCATED]"

    def log_usage(self, prompt_tokens: int, completion_tokens: int):
        """Thread-safe mechanism to log token expenditure directly from the API response payload."""
        # Acquire lock to ensure only one thread mutates global count simultaneously
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1

    def get_estimated_cost(self, model="deepseek-chat") -> float:
        """Calculates total accumulated USD cost projection via thread-safe reads."""
        # Lock to ensure no mid-write states are read
        with self._lock:
            # Fallback to deepseek pricing if the requested model isn't configured
            rates = PRICING.get(model, PRICING["deepseek-chat"])
            # Divide by 1 Million since prices are quoted per 1M tokens
            prompt_cost = (self.total_prompt_tokens / 1_000_000) * rates["prompt"]
            completion_cost = (self.total_completion_tokens / 1_000_000) * rates["completion"]
            return prompt_cost + completion_cost

    def print_summary(self, model="deepseek-chat"):
        """Outputs total API token usage footprint and cost metrics to the terminal."""
        cost = self.get_estimated_cost(model)
        
        # Safely fetch scalar usage details inside lock before printing
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
# 3. GLOBAL WRAPPER FUNCTIONS (Untuk diimpor oleh modul lain/Exported interface)
# ==============================================================================
# Initialize a global singleton instance of the TokenTracker class
# This ensures that token counts persist across multiple files and function calls imported into the same runtime
tracker = TokenTracker()

def count_tokens(text: str) -> int:
    # Wrap the instance method to provide a simple procedural call for external files
    return tracker.count_tokens(text)

def truncate_text(text: str, max_tokens: int = 2000) -> str:
    # Wrap the truncation instance method for external use
    return tracker.truncate_text(text, max_tokens)

def log_api_usage(prompt_tokens: int, completion_tokens: int):
    # Wrap the logging method, ensuring global state registers every thread's payload footprint
    tracker.log_usage(prompt_tokens, completion_tokens)

def print_token_summary(model="deepseek-chat"):
    # Output the final cost analysis to the terminal matching the global state
    tracker.print_summary(model)


# ==============================================================================
# DEBUG / LOCAL TESTING EXECUTION BLOCK
# ==============================================================================
# This block runs only if this file is executed directly (not when imported as a module)
if __name__ == "__main__":
    print("--- 🛠️ Menguji Modul Token Tracking ---")
    
    # 1. Test offline token calculations via tiktoken / string split fallback
    sample_text = "Proyek ini mengalami risiko pembengkakan anggaran karena kegagalan pengiriman vendor."
    tokens = count_tokens(sample_text)
    print(f"\nTeks: '{sample_text}'")
    print(f"Jumlah Token Estimasi (Offline): {tokens} tokens")
    
    # 2. Test the text truncation safeguard mechanisms
    long_text = "Risk Description: " + ("Error data " * 500)
    print(f"\nPanjang Teks Asli: {count_tokens(long_text)} tokens")
    safe_text = truncate_text(long_text, max_tokens=20)
    print(f"Hasil Truncate (max 20): {safe_text}")
    
    # 3. Test Thread-Safe global state Logging (Mocking API inputs)
    print("\n[Simulasi] Menerima balasan dari API sebanyak 3 kali (Multithreading)...")
    log_api_usage(prompt_tokens=1500, completion_tokens=50) # Simulation Call 1
    log_api_usage(prompt_tokens=1450, completion_tokens=45) # Simulation Call 2
    log_api_usage(prompt_tokens=1600, completion_tokens=60) # Simulation Call 3
    
    # 4. Trigger the final calculation output sequence
    print_token_summary()