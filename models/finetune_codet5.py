"""
PART 3c — CodeT5 FINE-TUNING (Member 3)
=========================================
Fine-tunes CodeT5-small on the DS Mentor dataset:
  Input:  natural language question + stage context
  Output: Python code snippet

Architecture:
  - Base model: Salesforce/codet5-small (60M params, feasible on CPU/Colab)
  - Fine-tune with seq2seq (encoder-decoder)
  - Training data: data/stage_labeled_dataset.csv (275 pairs)
  - Inference: generate_code_codet5(query, stage_name) → code string

Usage:
  # Train (needs ~20 min on GPU, ~2 hrs on CPU for small model)
  python models/finetune_codet5.py --train

  # Inference only (uses saved model or TF-IDF fallback)
  from models.finetune_codet5 import generate_codet5
  code = generate_codet5("How do I fill missing values in Age?", "Preprocessing")
"""
import csv, os, json, argparse
from pathlib import Path

MODEL_DIR   = "models/codet5_finetuned"
DATA_PATH   = "data/stage_labeled_dataset.csv"
BASE_MODEL  = "Salesforce/codet5-small"


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_finetune_data(path=DATA_PATH):
    """
    Load (input_text, target_code) pairs for seq2seq fine-tuning.
    Input format:  "stage: {stage_name} question: {explanation}"
    Target format: "{code}"
    """
    inputs, targets = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stage_name = row.get("stage_name", "")
            explanation = row.get("explanation", "")
            code = row.get("code", "")
            if not explanation or not code or len(code) < 10:
                continue
            # Format: task-prefix style for T5
            input_text = f"generate python: stage={stage_name} question={explanation}"
            inputs.append(input_text)
            targets.append(code)
    print(f"[INFO] Loaded {len(inputs)} training pairs from {path}")
    return inputs, targets


# ── Fine-tuning ───────────────────────────────────────────────────────────────
def finetune_codet5(
    data_path=DATA_PATH,
    output_dir=MODEL_DIR,
    epochs=5,
    batch_size=4,
    max_input_len=256,
    max_target_len=256,
    learning_rate=5e-5,
):
    """
    Fine-tune CodeT5-small on the DS Mentor QA pairs.

    Hardware requirements:
      - GPU (recommended): ~20 min on T4
      - CPU (feasible):    ~2 hours
      - Memory: ~4GB RAM minimum

    The model is saved to models/codet5_finetuned/ and loaded
    automatically when generate_codet5() is called.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            T5ForConditionalGeneration,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        from torch.utils.data import Dataset
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install: pip install transformers torch scikit-learn")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on: {device}")

    inputs, targets = load_finetune_data(data_path)
    X_tr, X_val, y_tr, y_val = train_test_split(
        inputs, targets, test_size=0.15, random_state=42
    )
    print(f"[INFO] Train: {len(X_tr)}, Val: {len(X_val)}")

    # use_fast=False: avoids HF/tokenizers crashes on some stacks (e.g. Kaggle) with CodeT5.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.to(device)

    class CodeDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs  = tokenizer(inputs,  max_length=max_input_len,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")
            self.targets = tokenizer(targets, max_length=max_target_len,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")
        def __len__(self): return len(self.inputs["input_ids"])
        def __getitem__(self, i):
            labels = self.targets["input_ids"][i].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "input_ids":      self.inputs["input_ids"][i],
                "attention_mask": self.inputs["attention_mask"][i],
                "labels":         labels,
            }

    train_ds = CodeDataset(X_tr, y_tr)
    val_ds   = CodeDataset(X_val, y_val)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),  # faster on GPU
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print(f"[INFO] Starting fine-tuning for {epochs} epochs...")
    trainer.train()

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "base_model":   BASE_MODEL,
        "epochs":       epochs,
        "train_size":   len(X_tr),
        "val_size":     len(X_val),
        "max_input":    max_input_len,
        "max_target":   max_target_len,
        "device":       device,
    }
    with open(f"{output_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] CodeT5 fine-tuned model saved → {output_dir}/")
    return True


# ── Inference ─────────────────────────────────────────────────────────────────
_codet5_model     = None
_codet5_tokenizer = None


def generate_codet5(query: str, stage_name: str = "",  # noqa
                    max_new_tokens: int = 200) -> dict:
    """
    Generate Python code for a data science query using fine-tuned CodeT5.

    Falls back to the template-based generator if CodeT5 model isn't trained.

    Args:
        query:      Natural language question.
        stage_name: Pipeline stage name for context (e.g. "Preprocessing").
        max_new_tokens: Max tokens to generate.

    Returns:
        {
          "code":   str,    # generated Python code
          "method": str,    # "codet5" | "template_fallback"
          "valid":  bool,   # syntax valid?
        }
    """
    global _codet5_model, _codet5_tokenizer

    # ── Try deployed/fine-tuned model directory first ─────────────────
    if Path(MODEL_DIR).exists():
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            if _codet5_model is None:
                print(f"[INFO] Loading CodeT5 from {MODEL_DIR}...")
                _codet5_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
                _codet5_model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
                _codet5_model.eval()

            input_text = f"generate python: stage={stage_name} question={query}"
            inputs = _codet5_tokenizer(
                input_text, return_tensors="pt",
                max_length=256, truncation=True
            )
            with torch.no_grad():
                outputs = _codet5_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            code = _codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Validate syntax
            import ast
            try:
                ast.parse(code)
                valid = True
            except SyntaxError:
                valid = False

            return {"code": code, "method": "codet5", "valid": valid}

        except Exception as e:
            print(f"[WARN] CodeT5 inference failed ({e}), using template fallback")

    # ── Fallback to base CodeT5 if no local model directory is present ─
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        if _codet5_model is None:
            _codet5_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
            _codet5_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
            _codet5_model.eval()

        input_text = f"generate python: stage={stage_name} question={query}"
        inputs = _codet5_tokenizer(
            input_text, return_tensors="pt",
            max_length=256, truncation=True
        )
        with torch.no_grad():
            outputs = _codet5_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        code = _codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        import ast
        try:
            ast.parse(code)
            valid = True
        except SyntaxError:
            valid = False

        return {"code": code, "method": "codet5_base", "valid": valid}
    except Exception:
        pass

    # ── Fallback: template-based generator ───────────────────────────
    from modules.code_generator import generate_code, validate_code_syntax
    from models.stage_classifier import extract_stage_num

    # Reverse map stage_name → stage_num
    name_to_num = {v: k for k, v in {
        1:"Problem Understanding", 2:"Data Loading",
        3:"Exploratory Data Analysis", 4:"Preprocessing",
        5:"Feature Engineering", 6:"Modeling", 7:"Evaluation",
    }.items()}
    stage_num = name_to_num.get(stage_name, 1)

    result = generate_code(query, stage_num)
    valid  = validate_code_syntax(result["code"])
    return {
        "code":   result["code"],
        "method": f"template_fallback ({result['method']})",
        "valid":  valid["valid"],
    }


def compute_code_metrics(generated: str, reference: str) -> dict:
    """
    Compute code quality metrics between generated and reference code.

    Metrics:
      - token_overlap:  Jaccard similarity on code tokens
      - line_match:     Fraction of reference lines appearing in generated
      - syntax_valid:   Whether generated code parses cleanly
      - length_ratio:   len(generated) / len(reference)
    """
    import ast

    # Syntax
    try:
        ast.parse(generated)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False

    # Tokenise (split on whitespace + common delimiters)
    import re
    def tokenise(code):
        return set(re.findall(r"[\w]+", code.lower()))

    gen_tokens = tokenise(generated)
    ref_tokens = tokenise(reference)
    intersection = gen_tokens & ref_tokens
    union = gen_tokens | ref_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    # Line match
    ref_lines  = set(l.strip() for l in reference.splitlines() if l.strip())
    gen_lines  = set(l.strip() for l in generated.splitlines() if l.strip())
    line_match = len(ref_lines & gen_lines) / len(ref_lines) if ref_lines else 0.0

    return {
        "syntax_valid":  syntax_ok,
        "token_overlap": round(jaccard, 4),
        "line_match":    round(line_match, 4),
        "length_ratio":  round(len(generated) / max(len(reference), 1), 3),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5 Fine-tuning for DS Mentor")
    parser.add_argument("--train",     action="store_true", help="Run fine-tuning")
    parser.add_argument("--demo",      action="store_true", help="Run inference demo")
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch",     type=int, default=4)
    args = parser.parse_args()

    if args.train:
        print("=== CodeT5 Fine-Tuning ===")
        success = finetune_codet5(epochs=args.epochs, batch_size=args.batch)
        if success:
            print("\n✅ Fine-tuning complete!")
        else:
            print("\n❌ Fine-tuning failed — check dependencies.")

    if args.demo or not args.train:
        print("\n=== CodeT5 Inference Demo ===")
        test_cases = [
            ("How do I fill missing values in Age column?",    "Preprocessing"),
            ("Train a Random Forest with 200 estimators",       "Modeling"),
            ("Plot a correlation heatmap",                      "Exploratory Data Analysis"),
            ("Compute AUC score and confusion matrix",          "Evaluation"),
            ("Encode the Sex column with LabelEncoder",         "Feature Engineering"),
        ]
        for query, stage in test_cases:
            result = generate_codet5(query, stage)
            print(f"\nQuery:  {query}")
            print(f"Stage:  {stage}")
            print(f"Method: {result['method']} | Syntax valid: {result['valid']}")
            print(f"Code:\n{result['code'][:300]}")
            print("─" * 50)
