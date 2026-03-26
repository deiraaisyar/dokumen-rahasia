from pathlib import Path

from submission_pipeline.extracting import main as run_extracting
from submission_pipeline.preprocessing import run_preprocessing
from submission_pipeline.generating_outputs import run_generating_outputs
from submission_pipeline.evaluating import run_evaluating


def _ensure_submission_dirs():
    """Create submission branch folder structure at repository root."""
    required_dirs = [
        "input",
        "inputs",
        "outputs",
        "ectracted_inputs",
        "extracted_inputs",
        "extracted_outputs",
        "generated_inputs",
        "generated_outputs",
        "preprocesed_inputs",
        "preprocessed_inputs",
        "preprocesed_outputs",
        "preprocessed_outputs",
    ]
    for dirname in required_dirs:
        Path(dirname).mkdir(parents=True, exist_ok=True)


def main():
    _ensure_submission_dirs()

    print("=== Submission Step 1: Extracting ===")
    run_extracting()

    print("\n=== Submission Step 2: Preprocessing ===")
    run_preprocessing()

    print("\n=== Submission Step 3: Generating Outputs ===")
    run_generating_outputs()

    print("\n=== Submission Step 4: Evaluating ===")
    run_evaluating()

    print("\nSubmission pipeline complete.")


if __name__ == "__main__":
    main()
