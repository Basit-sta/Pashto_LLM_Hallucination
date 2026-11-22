import json
import random
from pathlib import Path
import pandas as pd
import evaluate

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


def compute_automatic_metrics(records: list, rouge_metric, bertscore_metric) -> dict:
    preds = [r["model_answer"] for r in records]
    refs = [r["reference_answer"] for r in records]

    rouge_res = rouge_metric.compute(predictions=preds, references=refs)
    bert_res = bertscore_metric.compute(
        predictions=preds, references=refs, model_type="xlm-roberta-base"
    )

    # Abstention accuracy on unanswerable items (evidence_sufficient == "No")
    unanswerable = [r for r in records if r["evidence_sufficient"] == "No"]
    if unanswerable:
        correct_abstentions = sum(1 for r in unanswerable if check_abstention(r["model_answer"]))
        abstention_acc = correct_abstentions / len(unanswerable)
    else:
        abstention_acc = None

    # False refusal rate on answerable items (evidence_sufficient == "Yes")
    answerable = [r for r in records if r["evidence_sufficient"] == "Yes"]
    if answerable:
        false_refusals = sum(1 for r in answerable if check_abstention(r["model_answer"]))
        false_refusal_rate = false_refusals / len(answerable)
    else:
        false_refusal_rate = None

    return {
        "total_samples": len(records),
        "rouge1": round(rouge_res["rouge1"], 4),
        "rougeL": round(rouge_res["rougeL"], 4),
        "bertscore_f1": round(sum(bert_res["f1"]) / len(bert_res["f1"]), 4),
        "abstention_accuracy": round(abstention_acc, 4) if abstention_acc is not None else "N/A",
        "false_refusal_rate": round(false_refusal_rate, 4) if false_refusal_rate is not None else "N/A",
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
    base_records = load_jsonl(BASE_PREDS_FILE)
    ft_records = load_jsonl(FT_PREDS_FILE)

    print("Loading ROUGE and XLM-RoBERTa BERTScore metrics...")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    base_metrics = compute_automatic_metrics(base_records, rouge_metric, bertscore_metric)
    ft_metrics = compute_automatic_metrics(ft_records, rouge_metric, bertscore_metric)

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
