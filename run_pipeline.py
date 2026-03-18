from pipeline.extracting import main as run_extracting
from pipeline.preprocessing import run_preprocessing
from pipeline.generating_outputs import run_generating_outputs

def main():
    print("=== Step 1: Extracting ===")
    run_extracting()

    print("\n=== Step 2: Preprocessing ===")
    run_preprocessing()

    print("\n=== Step 3: Generating Outputs ===")
    run_generating_outputs()

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()