import json
import random
import re
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "pashto_qa_raw.jsonl"

random.seed(42)


def load_existing_records(filepath: Path) -> list:
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)  # Remove Wikipedia citation brackets [1]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_topic(text: str) -> str:
    """Assign each Pashto passage to one of the 5 proposal domains."""
    if any(k in text for k in ["ولایت", "سیند", "غر", "پلازمېنه", "مساحت", "بند", "سمندر", "جغرافیه", "ښار"]):
        return "Geography"
    if any(k in text for k in ["تاريخ", "تاریخ", "سلطان", "پاچا", "جګړه", "امپراتوري", "پېړۍ", "میلادي", "خپلواکي"]):
        return "History"
    if any(k in text for k in ["ساینس", "کیمیا", "فزیک", "بیولوژي", "سیاره", "لمر", "اوبه", "انرژي", "حجره", "تودوخه"]):
        return "Science"
    if any(k in text for k in ["شاعر", "ادب", "کتاب", "شعر", "کلتور", "دود", "موسیقي", "ژبه", "دیوان", "هنر"]):
        return "Culture"
    return "Public Information"


def generate_qa_from_pashto_wiki(target_new_records: int = 550, start_index: int = 76) -> list:
    """
    Fetch verified Pashto Wikipedia articles from Hugging Face (`wikimedia/wikipedia`, `20231101.ps`)
    and construct grounded QA pairs + 20% insufficient-evidence abstention pairs.
    """
    print("Streaming verified Pashto Wikipedia corpus (wikimedia/wikipedia -> 20231101.ps)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ps", split="train", streaming=True)

    passages = []
    for row in ds:
        title = clean_text(row.get("title", ""))
        text = clean_text(row.get("text", ""))
        # Keep clean, informative encyclopedic paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if 80 <= len(p.strip()) <= 450]
        if title and paragraphs and ":" not in title:
            passages.append((title, paragraphs))
        if len(passages) >= target_new_records + 100:
            break

    print(f"Collected {len(passages)} clean Pashto Wikipedia passages.")
    records = []
    idx = start_index

    for i in range(min(target_new_records, len(passages))):
        title, evidence = passages[i]
        topic = classify_topic(title + " " + evidence)
        rec_id = f"qa_{idx:03d}"

        # Make every 5th record (20%) a realistic abstention example ("evidence_sufficient": "No")
        if i % 5 == 4 and i + 1 < len(passages):
            other_title, _ = passages[i + 1]
            question = f"د «{other_title}» د دقیق تاسیس نېټه او کلنۍ بودیجه څومره ده؟"
            ref_answer = (
                f"د ورکړل شویو معلوماتو پر بنسټ، د «{other_title}» په اړه په دې متن کې کافي شواهد نشته."
            )
            sufficient = "No"
        else:
            question = f"د ورکړل شویو معلوماتو له مخې، «{title}» څه شی دی یا کومه ځانګړتیا لري؟"
            # Use the verified first 1-2 sentences as concise ground-truth reference
            sentences = [s.strip() for s in evidence.split("।") if s.strip()]
            if len(sentences) == 1:
                sentences = [s.strip() for s in evidence.split(".") if s.strip()]
            ref_answer = (sentences + ".") if sentences else evidence
            sufficient = "Yes"

        records.append({
            "id": rec_id,
            "topic": topic,
            "question": question,
            "evidence": evidence,
            "reference_answer": ref_answer,
            "evidence_sufficient": sufficient,
            "source": f"Pashto Wikipedia ({title})",
        })
        idx += 1

    return records


def main():
    existing = load_existing_records(RAW_FILE)
    print(f"Loaded {len(existing)} existing verified records from {RAW_FILE.name}.")

    new_records = generate_qa_from_pashto_wiki(
        target_new_records=550, start_index=len(existing) + 1
    )

    all_records = existing + new_records
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for item in all_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    yes_count = sum(1 for r in all_records if r["evidence_sufficient"] == "Yes")
    no_count = sum(1 for r in all_records if r["evidence_sufficient"] == "No")
    print(f"\nSaved {len(all_records)} total records -> {RAW_FILE}")
    print(f"Answerable (Yes): {yes_count} | Abstention (No): {no_count}")


if __name__ == "__main__":
    main()
