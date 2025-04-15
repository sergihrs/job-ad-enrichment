import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.data import load_data_mc
from peft import LoraConfig, get_peft_model, TaskType


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
    train_dataset, test_dataset = load_data_mc(dataset)

    # Tokenize the datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["job_text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Remove the original "job_text" column
    train_dataset = train_dataset.remove_columns(["job_text"])
    test_dataset = test_dataset.remove_columns(["job_text"])

    # Set the datasets format to PyTorch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Determine the number of unique labels
    train_labels = train_dataset["labels"]
    test_labels = test_dataset["labels"]
    unique_labels = set(train_labels) | set(test_labels)
    num_labels = len(unique_labels)

    if train_dataset.features["labels"].dtype != int:
        # If labels are strings, convert them to integers
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        train_dataset = train_dataset.map(
            lambda x: {"labels": label_to_id[x["labels"]]},
            remove_columns=["labels"],
        )
        test_dataset = test_dataset.map(
            lambda x: {"labels": label_to_id[x["labels"]]},
            remove_columns=["labels"],
        )
        print(f"Labels converted to integers. Mapping: {label_to_id}")

    # Load the Pretrained Model with a Classification Head
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
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
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        eval_strategy="epoch",  # evaluate at the end of each epoch
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-5,  # learning rate
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,  # log every 10 steps
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,  # ensure the tokenizer is passed for proper data collation
    )

    # 6. Start Training
    trainer.train()

    # 7. Save the Finetuned Model and Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
