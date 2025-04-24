from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoTokenizer,
)

from src.data import load_data_mc
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import accuracy_score
import numpy as np
import torch


def train(
    dataset: str = "seniority", output_dir: str = "./models/distilbert-finetuned"
):
    """
    Main function to train the DistilBERT model for multiclass classification.

    This function loads the dataset, initializes the tokenizer, and trains the model.
    The csv files should have a job_text column and a labels column.

    Args:
        dataset (str): The dataset to use.
        output_dir (str): The directory to save the model checkpoints.
    """
    # Load the dataset
    train_dataset, test_dataset, id_to_label, label_to_id = load_data_mc(dataset)

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["job_text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    #  Collate the data
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set the datasets format to PyTorch tensors
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load the Pretrained Model with a Classification Head
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_to_id),  # number of labels in the dataset
        id2label=id_to_label,  # mapping from label id to label name
        label2id=label_to_id,  # mapping from label name to label id
    )
    # Wrap the model with LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # sequence classification task
        inference_mode=False,  # training mode
        r=8,  # LoRA rank
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,  # dropout to regularize training
        target_modules=["q_lin", "v_lin"],  # target modules to apply LoRA to
    )
    # Wrap the original model with PEFT to create a LoRA model.
    model = get_peft_model(model, lora_config)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # outdir for checkpoints
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_strategy="steps",  # evaluation strategy to adopt during training
        eval_steps=10,  # evaluate every 10 steps
        save_strategy="best",  # save checkpoint every eval_steps
        load_best_model_at_end=True,  # load the best model at the end of training
        learning_rate=1e-4,  # learning rate
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=1,  # log every 10 steps
        metric_for_best_model="loss",  # metric to use for best model selection
        greater_is_better=False,
        label_names=["labels"],  # specify the label column name
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,  # use the collator for padding
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # ensure the tokenizer is passed for proper data collation
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ],  # early stopping callback
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
        },  # compute accuracy
    )

    # 6. Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")

    # 7. Save the Finetuned Model and Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(
    dataset: str = "seniority",
    checkpoint_path: str = "./models/distilbert-finetuned/checkpoint-168",
):
    """
    Load a finetuned DistilBERT checkpoint and measure accuracy on the held‑out
    test set that `load_data_mc()` returns.

    Args:
        dataset (str):  Name of the dataset to load via `load_data_mc`.
        checkpoint_path (str):  Directory of the saved checkpoint to evaluate.
    """
    ## 1. Load data split -----------------------------------------------------
    _, test_dataset = load_data_mc(dataset)

    ## 2. Load tokenizer & model from the checkpoint --------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_path)
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)

    ## 3. Tokenize the test set (same recipe you used for training) -----------
    def tokenize_function(examples):
        return tokenizer(
            examples["job_text"],
            padding="max_length",
            truncation=True,
            max_length=512,  # keep in sync with training defaults
        )

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.remove_columns(["job_text"])
    test_dataset.set_format("torch")

    ## 4. Run predictions -----------------------------------------------------
    trainer = Trainer(model=model, tokenizer=tokenizer)

    model.eval()  # make sure dropout etc. are off
    with torch.no_grad():
        preds_output = trainer.predict(test_dataset)

    logits = preds_output.predictions  # shape: (N, num_labels)
    preds = np.argmax(logits, axis=-1)
    labels = preds_output.label_ids

    ## 5. Compute & print accuracy -------------------------------------------
    acc = accuracy_score(labels, preds)
    print(f"Accuracy on test set: {acc:.4%}")

    return acc


if __name__ == "__main__":
    train()
    # evaluate()
