from pipeline.extracting import main as run_extracting
from pipeline.preprocessing import run_preprocessing
from pipeline.evaluating import run_evaluating

def main():
    print("=== Step 1: Extracting ===")
    run_extracting()

    print("\n=== Step 2: Preprocessing ===")
    run_preprocessing()

    print("\n=== Step 3: Evaluating ===")
    run_evaluating()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()