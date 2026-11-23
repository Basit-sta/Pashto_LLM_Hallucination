import argparse
import json
import random
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
BASE_PREDS_FILE = RESULTS_DIR / "baseline_predictions.jsonl"
FT_PREDS_FILE = RESULTS_DIR / "finetuned_predictions.jsonl"


def load_jsonl(filepath: Path) -> list:
    if not filepath.exists():
        raise FileNotFoundError(f"Prediction file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_abstention(text: str) -> bool:
    """Detect whether the model appropriately abstained in Pashto."""
    abstention_phrases = [
        "کافي شواهد نشته",
        "کافی شواهد نشته",
        "کافي معلومات نشته",
        "کافی معلومات نشته",
        "معلومات نشته",
        "نه دي ذکر شوي",
        "نه دی ذکر شوی",
        "نه توضیح دی",
    ]
    return any(phrase in text for phrase in abstention_phrases)


def pashto_token_f1(pred: str, ref: str) -> float:
    """Compute token-level F1 overlap preserving Pashto Unicode words."""
    p_tokens = pred.split()
    r_tokens = ref.split()
    if not p_tokens or not r_tokens:
        return 0.0
    common = set(p_tokens) & set(r_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(r_tokens)
    return 2 * prec * rec / (prec + rec)


def pashto_rouge_l(pred: str, ref: str) -> float:
    """Compute token-level LCS F1 (ROUGE-L) preserving Pashto Unicode tokens."""
    p_tokens = pred.split()
    r_tokens = ref.split()
    if not p_tokens or not r_tokens:
        return 0.0
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def compute_automatic_metrics(records: list, use_bertscore: bool = False) -> dict:
    preds = [r["model_answer"] for r in records]
    refs = [r["reference_answer"] for r in records]

    rouge_l_scores = [pashto_rouge_l(p, r) for p, r in zip(preds, refs)]
    token_f1_scores = [pashto_token_f1(p, r) for p, r in zip(preds, refs)]

    bert_f1 = "Skipped (use --bertscore)"
    if use_bertscore:
        from bert_score import score as bert_score_fn
        _, _, f1_tensor = bert_score_fn(
            preds, refs, model_type="xlm-roberta-base", verbose=False
        )
        bert_f1 = round(f1_tensor.mean().item(), 4)

    unanswerable = [r for r in records if r["evidence_sufficient"] == "No"]
    if unanswerable:
        correct_abstentions = sum(1 for r in unanswerable if check_abstention(r["model_answer"]))
        abstention_acc = round(correct_abstentions / len(unanswerable), 4)
    else:
        abstention_acc = "N/A"

    answerable = [r for r in records if r["evidence_sufficient"] == "Yes"]
    if answerable:
        false_refusals = sum(1 for r in answerable if check_abstention(r["model_answer"]))
        false_refusal_rate = round(false_refusals / len(answerable), 4)
    else:
        false_refusal_rate = "N/A"

    return {
        "total_samples": len(records),
        "pashto_token_f1": round(sum(token_f1_scores) / len(token_f1_scores), 4),
        "pashto_rougeL": round(sum(rouge_l_scores) / len(rouge_l_scores), 4),
        "bertscore_f1": bert_f1,
        "abstention_accuracy": abstention_acc,
        "false_refusal_rate": false_refusal_rate,
    }


def generate_blind_human_eval_sheet(base_records: list, ft_records: list, out_path: Path) -> None:
    """Create a blinded spreadsheet matching Appendix A of the research proposal."""
    rng = random.Random(42)
    rows = []
    key_rows = []

    for b, ft in zip(base_records, ft_records):
        pair = [("base", b["model_answer"]), ("finetuned", ft["model_answer"])]
        rng.shuffle(pair)

        for idx, (cond_name, ans_text) in enumerate(pair, start=1):
            anon_id = f"{b['id']}_M{idx}"
            rows.append({
                "Eval_ID": anon_id,
                "Topic": b["topic"],
                "Evidence_Sufficient": b["evidence_sufficient"],
                "Question_Pashto": b["question"],
                "Evidence_Pashto": b["evidence"],
                "Reference_Answer": b["reference_answer"],
                "Model_Response": ans_text,
                "Claim_Support (Supported/Contradicted/Unsupported)": "",
                "Fluency (1-5)": "",
                "Grammar_Correctness (1-5)": "",
                "Relevance (1-5)": "",
                "Completeness (1-5)": "",
                "Uncertainty_Handling (Appropriate/Inappropriate)": "",
                "Evaluator_Notes": "",
            })
            key_rows.append({
                "Eval_ID": anon_id,
                "Question_ID": b["id"],
                "True_Condition": cond_name,
            })

    df_blind = pd.DataFrame(rows)
    df_key = pd.DataFrame(key_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_blind.to_excel(writer, sheet_name="Blind_Annotation_Sheet", index=False)
        df_key.to_excel(writer, sheet_name="Answer_Key_Hidden", index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Pashto QA models")
    parser.add_argument("--bertscore", action="store_true", help="Also compute XLM-RoBERTa BERTScore")
    args = parser.parse_args()

    base_records = load_jsonl(BASE_PREDS_FILE)
    ft_records = load_jsonl(FT_PREDS_FILE)

    base_metrics = compute_automatic_metrics(base_records, use_bertscore=args.bertscore)
    ft_metrics = compute_automatic_metrics(ft_records, use_bertscore=args.bertscore)

    summary_df = pd.DataFrame([
        {"Condition": "Condition A (Base Model)", **base_metrics},
        {"Condition": "Condition B (QLoRA Fine-Tuned)", **ft_metrics},
    ])

    summary_csv = RESULTS_DIR / "automatic_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n=== Automatic Evaluation Summary ===")
    print(summary_df.to_string(index=False))

    eval_xlsx = RESULTS_DIR / "human_evaluation_sheet.xlsx"
    generate_blind_human_eval_sheet(base_records, ft_records, eval_xlsx)
    print(f"\nSaved blinded human evaluation workbook -> {eval_xlsx}")


if __name__ == "__main__":
    main()
