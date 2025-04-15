import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.data import load_data_mlc


def train(file_path: str = "data/", output_dir: str = "./models/distilbert-finetuned"):
    """
    Main function to train the DistilBERT model for multiclass classification.

    This function loads the dataset, initializes the tokenizer, and trains the model.
    The csv files should have a text column and a labels column.

    Args:
        file_path (str): The path to the dataset.
        output_dir (str): The directory to save the model checkpoints.
    """
    # Load the dataset
    train_dataset, test_dataset = load_data_mlc(file_path)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Optionally remove the original "text" column if you no longer need it.
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    # Set the datasets format to PyTorch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Determine the number of unique labels (assumes labels are numeric)
    train_labels = train_dataset["labels"]
    num_labels = len(set(train_labels.tolist()))

    # 3. Load the Pretrained Model with a Classification Head
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # outdir for checkpoints
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        evaluation_strategy="epoch",  # evaluate at the end of each epoch
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-5,  # learning rate
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,  # log every 10 steps
    )
    # 5. Initialize the Trainer
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
