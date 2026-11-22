# Reducing Hallucinations in Large Language Models for the Pashto Language Through Fine-Tuning

**Author:** Abdul Basit Stanikzai  
**Supervisor:** Dr. Abdul Malik  
**Institution:** Afghan International Islamic University (AIIU)  

## Project Overview
This experimental study investigates whether parameter-efficient supervised fine-tuning (LoRA / QLoRA) reduces factual hallucinations and improves abstention behavior in open-weight multilingual LLMs for Pashto question answering.

## Project Structure
- `data/raw/`: Verified Pashto QA pairs with supporting evidence.
- `data/processed/`: Cleaned and instruction-formatted Pashto datasets.
- `data/train_val_test/`: 70% train, 15% validation, and 15% held-out test splits.
- `notebooks/`: Interactive exploration, baseline testing, and evaluation.
- `src/`: Modular scripts for dataset preparation, training, and evaluation.
- `models/fine_tuned/`: Saved LoRA/QLoRA adapter checkpoints.
- `results/`: Evaluation metrics, human annotation sheets, and plots.
