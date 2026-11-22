import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "pashto_qa_raw.jsonl"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLIT_DIR = BASE_DIR / "data" / "train_val_test"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_pashto(text: str) -> str:
    """Clean whitespace and standardize Arabic Kaf/Yeh variants to standard Pashto."""
    if not text:
        return ""
    text = text.replace("\u0643", "\u06a9")  # Arabic Kaf (ك) -> Standard Kaf (ک)
    text = text.replace("\u064a", "\u06cc")  # Arabic Yeh (ي) -> Standard Yeh (ی)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_instruction_prompt(record: dict) -> dict:
    """Format each QA pair into a strict evidence-grounded Pashto instruction template."""
    system_prompt = (
        "تاسو یو دقیق او صادق پښتو مرستیال یاست. یوازې د ورکړل شویو شواهدو پر بنسټ پوښتنې ته لنډ او سم ځواب ورکړئ. "
        "که ورکړل شوي معلومات د ځواب لپاره کافي نه وي، نو له ځانه معلومات مه جوړوئ او په روښانه توګه ووایاست چې کافي شواهد نشته."
    )

    user_prompt = (
        f"شواهد (Evidence):\n{record['evidence']}\n\n"
        f"پوښتنه (Question):\n{record['question']}"
    )

    return {
        "id": record["id"],
        "topic": record.get("topic", "General"),
        "evidence_sufficient": record.get("evidence_sufficient", "Yes"),
        "source": record.get("source", "Unknown"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": record["reference_answer"]},
        ],
        "question": record["question"],
        "evidence": record["evidence"],
        "reference_answer": record["reference_answer"],
    }


def save_jsonl(records: list, filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_FILE}")

    raw_records = []
    seen_questions = set()

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["question"] = normalize_pashto(item["question"])
            item["evidence"] = normalize_pashto(item["evidence"])
            item["reference_answer"] = normalize_pashto(item["reference_answer"])

            if item["question"] in seen_questions:
                print(f"Skipping duplicate question at line {line_num}: {item['id']}")
                continue
            seen_questions.add(item["question"])
            raw_records.append(format_instruction_prompt(item))

    save_jsonl(raw_records, PROCESSED_DIR / "pashto_qa_formatted.jsonl")
    print(f"Total clean unique records: {len(raw_records)}")

    if len(raw_records) < 10:
        print("Dataset has fewer than 10 records (pilot mode). Saving all to train/val/test for pipeline testing.")
        save_jsonl(raw_records, SPLIT_DIR / "train.jsonl")
        save_jsonl(raw_records, SPLIT_DIR / "val.jsonl")
        save_jsonl(raw_records, SPLIT_DIR / "test.jsonl")
        return

    # 70% Train, 15% Validation, 15% Test split stratified by topic & sufficiency
    stratify_labels = [f"{r['topic']}_{r['evidence_sufficient']}" for r in raw_records]
    try:
        train_data, temp_data = train_test_split(
            raw_records, test_size=0.30, random_state=42, stratify=stratify_labels
        )
        temp_labels = [f"{r['topic']}_{r['evidence_sufficient']}" for r in temp_data]
        val_data, test_data = train_test_split(
            temp_data, test_size=0.50, random_state=42, stratify=temp_labels
        )
    except ValueError:
        train_data, temp_data = train_test_split(raw_records, test_size=0.30, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

    save_jsonl(train_data, SPLIT_DIR / "train.jsonl")
    save_jsonl(val_data, SPLIT_DIR / "val.jsonl")
    save_jsonl(test_data, SPLIT_DIR / "test.jsonl")

    print(f"Saved splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()
