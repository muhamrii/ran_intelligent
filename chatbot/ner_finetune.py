import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# --------------------------------------------------
# Fine-tune NER Model
# --------------------------------------------------
def fine_tune_ner_model(base_model: str, dataset_path: str, output_dir: str):
    """
    Fine-tune a pre-trained NER model on a custom dataset.

    Args:
        base_model (str): The name of the pre-trained model.
        dataset_path (str): Path to the dataset in Hugging Face format.
        output_dir (str): Directory to save the fine-tuned model.
    """
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=len(dataset["train"].features["ner_tags"].feature.names))

    # Tokenize dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    BASE_MODEL = "bert-base-cased"
    DATASET_PATH = "path/to/ner_dataset.json"  # Replace with your dataset path
    OUTPUT_DIR = "./fine_tuned_ner_model"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fine_tune_ner_model(BASE_MODEL, DATASET_PATH, OUTPUT_DIR)
