import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel based on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _normalize_text(value) -> str:
    """Normalize text for stable comparison across formatting differences."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _token_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity on token frequency vectors without external deps."""
    a = _normalize_text(text_a)
    b = _normalize_text(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    ca = Counter(a.split())
    cb = Counter(b.split())
    vocab = set(ca) | set(cb)
    dot = sum(ca[t] * cb[t] for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _to_float(value):
    """Convert values to float safely for numeric evaluation."""
    try:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _align_rows(gen_df: pd.DataFrame, ref_df: pd.DataFrame, key_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align generated and reference rows by key column; fallback to row order when needed."""
    if key_column in gen_df.columns and key_column in ref_df.columns:
        gen = gen_df.copy()
        ref = ref_df.copy()
        gen["__k"] = gen[key_column].astype(str).str.strip()
        ref["__k"] = ref[key_column].astype(str).str.strip()
        merged = gen.merge(ref, on="__k", how="inner", suffixes=("_gen", "_ref"))
        if len(merged) > 0:
            gen_cols = [c for c in merged.columns if c.endswith("_gen")]
            ref_cols = [c for c in merged.columns if c.endswith("_ref")]
            gen_aligned = merged[gen_cols].copy()
            ref_aligned = merged[ref_cols].copy()
            gen_aligned.columns = [c[:-4] for c in gen_aligned.columns]
            ref_aligned.columns = [c[:-4] for c in ref_aligned.columns]
            return gen_aligned.reset_index(drop=True), ref_aligned.reset_index(drop=True)

    n = min(len(gen_df), len(ref_df))
    return gen_df.iloc[:n].reset_index(drop=True), ref_df.iloc[:n].reset_index(drop=True)


def _evaluate_pair(doc_name: str, gen_path: Path, ref_path: Path, key_column: str) -> dict:
    """Evaluate one generated doc against its reference preprocessed output."""
    gen_df = _read_table(gen_path)
    ref_df = _read_table(ref_path)
    gen_df, ref_df = _align_rows(gen_df, ref_df, key_column)

    common_cols = [c for c in gen_df.columns if c in ref_df.columns]
    if not common_cols:
        return {
            "doc": doc_name,
            "rows_compared": 0,
            "text_cosine_avg": 0.0,
            "exact_match_rate": 0.0,
            "numeric_mae": None,
            "numeric_within_0_1_rate": None,
            "overall_score": 0.0,
        }

    text_scores = []
    exact_hits = 0
    exact_total = 0
    num_abs_errors = []
    num_within = 0
    num_total = 0

    for col in common_cols:
        gen_series = gen_df[col]
        ref_series = ref_df[col]

        gen_num = gen_series.apply(_to_float)
        ref_num = ref_series.apply(_to_float)
        numeric_mask = gen_num.notna() & ref_num.notna()

        if numeric_mask.any():
            for gv, rv in zip(gen_num[numeric_mask], ref_num[numeric_mask]):
                err = abs(gv - rv)
                num_abs_errors.append(err)
                if err <= 0.1:
                    num_within += 1
                num_total += 1
        else:
            for gv, rv in zip(gen_series, ref_series):
                gtxt = _normalize_text(gv)
                rtxt = _normalize_text(rv)
                text_scores.append(_token_cosine_similarity(gtxt, rtxt))
                exact_hits += int(gtxt == rtxt)
                exact_total += 1

    text_cosine_avg = sum(text_scores) / len(text_scores) if text_scores else 1.0
    exact_match_rate = exact_hits / exact_total if exact_total else 1.0
    numeric_mae = sum(num_abs_errors) / len(num_abs_errors) if num_abs_errors else None
    numeric_within_rate = num_within / num_total if num_total else None

    if numeric_mae is None:
        numeric_score = 1.0
    else:
        numeric_score = max(0.0, 1.0 - (numeric_mae / 10.0))

    overall_score = 0.5 * text_cosine_avg + 0.25 * exact_match_rate + 0.25 * numeric_score

    return {
        "doc": doc_name,
        "rows_compared": len(gen_df),
        "text_cosine_avg": round(text_cosine_avg, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "numeric_mae": None if numeric_mae is None else round(numeric_mae, 4),
        "numeric_within_0_1_rate": None if numeric_within_rate is None else round(numeric_within_rate, 4),
        "overall_score": round(overall_score, 4),
    }


def run_evaluating():
    """Run evaluation for generated doc1-3 against preprocessed reference outputs."""
    mappings = [
        {
            "doc": "doc1",
            "gen": Path("./generated_outputs/1. IVC DOE (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/1. IVC DOE (Final).csv"),
            "key": "Risk ID",
        },
        {
            "doc": "doc2",
            "gen": Path("./generated_outputs/2. City of York Council (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/2. City of York Council (Final).csv"),
            "key": "Risk ID",
        },
        {
            "doc": "doc3",
            "gen": Path("./generated_outputs/3. Digital Security IT Sample Register (Final).xlsx"),
            "ref": Path("./preprocessed_outputs/3. Digital Security IT Sample Register (Final).csv"),
            "key": "Number",
        },
    ]

    results = []
    for item in mappings:
        if not item["gen"].exists() or not item["ref"].exists():
            print(f"Skip {item['doc']}: missing file")
            continue
        result = _evaluate_pair(item["doc"], item["gen"], item["ref"], item["key"])
        results.append(result)
        print(
            f"{item['doc']} | rows={result['rows_compared']} | "
            f"cos={result['text_cosine_avg']} | exact={result['exact_match_rate']} | "
            f"mae={result['numeric_mae']} | overall={result['overall_score']}"
        )

    if not results:
        print("No evaluation results generated.")
        return

    out_path = Path("./generated_outputs/evaluation_doc1_doc3.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Evaluation saved → {out_path}")