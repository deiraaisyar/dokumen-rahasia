# dokumen-rahasia

Small pipeline to process several risk register documents (Excel + PDF) into a consistent output format.

## Overview

This project runs in 3 steps:

1. **Extracting**  
   Reads files from `data/inputs/` and converts them into CSV files in `data/extracted_inputs/`.

2. **Preprocessing**  
   Cleans text fields and writes results to `data/preprocessed_inputs/`.

3. **Generating outputs**  
   Uses Groq LLM calls to infer missing fields and writes final files to `data/generated_outputs/`.

Main entry point: `run_pipeline.py`.

## Requirements

- Python 3.10+ recommended
- Input files placed in `data/inputs/`
- API keys in `.env` (see `.env.example`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment setup

Copy env template:

```bash
cp .env.example .env
```

Then fill:

- `GROQ_API_KEY`

## How to run

Run full pipeline:

```bash
python run_pipeline.py
```

Run step-by-step (optional):

```bash
python pipeline/extracting.py
python pipeline/preprocessing.py
python pipeline/generating_outputs.py
```

## Output folders

- `data/extracted_inputs/` → extracted raw CSVs
- `data/preprocessed_inputs/` → cleaned CSVs
- `data/generated_outputs/` → final generated CSVs