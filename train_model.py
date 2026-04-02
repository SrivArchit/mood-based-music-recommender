"""
Fine-tune DistilBERT on GoEmotions dataset for multi-label emotion classification.
Converted from notebooks/bertFineTunning.ipynb for local execution.

Usage: python train_model.py
Output: ./fine-tuned-goemotions-model/
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    EvalPrediction
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# --- Configuration ---
os.environ["WANDB_DISABLED"] = "true"
MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = "./fine-tuned-goemotions-model"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

def main():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # --- 1. Load Dataset ---
    print("\n📦 Loading GoEmotions dataset from HuggingFace...")
    dataset = load_dataset("go_emotions", "simplified")
    
    label_names = dataset["train"].features["labels"].feature.names
    num_labels = len(label_names)
    print(f"   Found {num_labels} emotion labels: {label_names}")
    
    # --- 2. Load Tokenizer ---
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- 3. Preprocessing ---
    def tokenize_data(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    def create_float_labels(examples):
        labels_one_hot = []
        for labels_list in examples["labels"]:
            one_hot_vector = [0.0] * num_labels
            for label_id in labels_list:
                if 0 <= label_id < num_labels:
                    one_hot_vector[label_id] = 1.0
            labels_one_hot.append(one_hot_vector)
        examples["float_labels"] = labels_one_hot
        return examples
    
    print("\n⚙️  Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    
    print("⚙️  Creating one-hot labels...")
    tokenized_dataset = tokenized_dataset.map(create_float_labels, batched=True)
    
    # Remove old columns, rename float_labels -> labels
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "id", "labels"])
    tokenized_dataset = tokenized_dataset.rename_column("float_labels", "labels")
    tokenized_dataset.set_format("torch")
    
    # --- 4. Model Initialization ---
    print(f"\n🧠 Loading model: {MODEL_NAME} ({num_labels} labels)")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # --- 5. Metrics ---
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = (probs.numpy() > 0.5).astype(int)
        y_true = labels.astype(int)
        
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        try:
            roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        except ValueError:
            roc_auc = 0.0
        accuracy = accuracy_score(y_true, y_pred)
        
        return {'f1_micro': f1, 'roc_auc': roc_auc, 'accuracy': accuracy}
    
    # --- 6. Training Arguments ---
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        fp16=use_fp16,
        logging_dir='./logs',
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=1,
    )
    
    # --- 7. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # --- 8. Train! ---
    print(f"\n🚀 Starting training ({NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...")
    print(f"   FP16: {use_fp16}")
    print(f"   Train samples: {len(tokenized_dataset['train'])}")
    print(f"   Validation samples: {len(tokenized_dataset['validation'])}")
    print("-" * 60)
    
    trainer.train()
    
    # --- 9. Evaluate ---
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("\nTest Set Results:")
    for key, value in test_results.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # --- 10. Save Model ---
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    trainer.save_model(MODEL_SAVE_PATH)
    # Also save tokenizer for completeness
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print(f"\n✅ Done! Model saved to: {MODEL_SAVE_PATH}")
    print("   You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()
