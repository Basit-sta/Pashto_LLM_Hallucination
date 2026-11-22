import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "data" / "train_val_test"
MODEL_OUT_DIR = BASE_DIR / "models" / "fine_tuned" / "pashto_qlora_adapter"
RESULTS_DIR = BASE_DIR / "results"

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(filepath: Path) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list, filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_quantized_model_and_tokenizer(model_name: str):
    """Load 4-bit QLoRA base model and tokenizer for T4 GPU compatibility."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer


def generate_predictions(model, tokenizer, test_records: list, condition_label: str) -> list:
    """Run deterministic inference on the held-out test set."""
    model.eval()
    predictions = []

    for rec in test_records:
        # Use only system + user messages for inference
        prompt_messages = [m for m in rec["messages"] if m["role"] in ("system", "user")]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,  # Deterministic decoding per Section 10.4
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append({
            "id": rec["id"],
            "topic": rec["topic"],
            "evidence_sufficient": rec["evidence_sufficient"],
            "question": rec["question"],
            "evidence": rec["evidence"],
            "reference_answer": rec["reference_answer"],
            "condition": condition_label,
            "model_answer": response_text,
        })

    return predictions


def tokenize_for_sft(records: list, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize full chat conversations and mask prompt tokens so loss is computed only on answers."""
    processed = []
    for rec in records:
        prompt_messages = [m for m in rec["messages"] if m["role"] in ("system", "user")]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
        full_tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        labels = labels[:len(input_ids)]

        processed.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(description="Baseline & QLoRA Fine-Tuning for Pashto QA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_records = load_jsonl(SPLIT_DIR / "train.jsonl")
    val_records = load_jsonl(SPLIT_DIR / "val.jsonl")
    test_records = load_jsonl(SPLIT_DIR / "test.jsonl")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_quantized_model_and_tokenizer(args.model_name)

    # 1. Condition A: Baseline evaluation before fine-tuning
    print("Running Condition A (Base Model) on test set...")
    base_preds = generate_predictions(model, tokenizer, test_records, condition_label="base")
    save_jsonl(base_preds, RESULTS_DIR / "baseline_predictions.jsonl")
    print(f"Saved baseline predictions -> {RESULTS_DIR / 'baseline_predictions.jsonl'}")

    # 2. Prepare model for QLoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = tokenize_for_sft(train_records, tokenizer)
    val_ds = tokenize_for_sft(val_records, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    print("Starting Condition B (QLoRA Fine-Tuning)...")
    trainer.train()

    # Save LoRA adapter and tokenizer
    model.save_pretrained(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"Saved fine-tuned adapter -> {MODEL_OUT_DIR}")

    # 3. Condition B: Fine-tuned evaluation on the exact same test set
    print("Running Condition B (Fine-Tuned Model) on test set...")
    ft_preds = generate_predictions(model, tokenizer, test_records, condition_label="finetuned")
    save_jsonl(ft_preds, RESULTS_DIR / "finetuned_predictions.jsonl")
    print(f"Saved fine-tuned predictions -> {RESULTS_DIR / 'finetuned_predictions.jsonl'}")


if __name__ == "__main__":
    main()
