from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from src.data_seniority import load_data_mc
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import torch.nn as nn


class WeightedTrainer(Trainer):
    """
    Overrides the Trainer class to use class weights for loss computation.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # register as buffer so it is moved to the right device automatically
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def show_trainable(model):
    """
    Show the number of trainable parameters in the model.
    """
    print("\n=== trainable parameters ===")
    total, trainable = 0, 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"{n:60s}  {p.numel():>8,d}")
    print(f"\ntrainable / total params: {trainable:,} / {total:,}")


def confirm_learning(trainer, model):
    """
    Check if the model is learning by checking the gradients of the parameters.
    """
    batch = next(iter(trainer.get_train_dataloader()))
    model.zero_grad()

    # forward + backward
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # check grad norms on your head
    for name, p in model.named_parameters():
        if p.requires_grad and ("classifier" in name or "pre_classifier" in name):
            print(f"{name:50s} grad norm = {p.grad.norm().item():.3e}")


def show_modules(model, depth=2):
    """
    Show the modules in the model. The depth parameter controls how many levels of modules to show.
    """
    print("\n=== modules ===")
    for n, m in model.named_modules():
        if n.count(".") <= depth:
            print(f"{n:50s}  -> {m.__class__.__name__}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


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

    # Sequence classification task
    pretrain_model_name = "distilbert-base-uncased"

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

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
        pretrain_model_name,
        num_labels=len(label_to_id),  # number of labels in the dataset
        id2label=id_to_label,  # mapping from label id to label name
        label2id=label_to_id,  # mapping from label name to label id
    )

    # Wrap the model with LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # sequence classification task
        inference_mode=False,  # training mode
        r=4,  # LoRA rank
        lora_alpha=32,  # scaling factor
        lora_dropout=0.05,  # dropout to regularize training
        target_modules=["q_lin", "v_lin"],  # target modules to apply LoRA to
        modules_to_save=[
            "classifier",
            "pre_classifier",
        ],  # modules to save during training
    )
    # Wrap the original model with PEFT to create a LoRA model.
    model = get_peft_model(model, lora_config)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # outdir for checkpoints
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        eval_strategy="steps",  # evaluation strategy to adopt during training
        eval_steps=15,  # evaluate every 10 steps
        save_strategy="steps",  # save checkpoint every eval_steps
        save_steps=15,  # save checkpoint every 10 steps
        load_best_model_at_end=True,  # load the best model at the end of training
        learning_rate=1e-4,  # learning rate
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=1,  # log every 10 steps
        metric_for_best_model="accuracy",  # metric to use for best model selection
        greater_is_better=True,
        label_names=["labels"],  # specify the label column name
    )

    label_to_weight = {
        "experienced": 130.5 / 204,
        "intermediate": 130.5 / 145,
        "entry level": 130.5 / 102,
        "senior": 130.5 / 71,
    }
    weights = np.array(
        [label_to_weight[id_to_label[i]] for i in range(len(label_to_id))]
    )

    # Initialize the Trainer
    trainer = WeightedTrainer(
        class_weights=torch.from_numpy(weights).float(),
        model=model,
        args=training_args,
        data_collator=data_collator,  # use the collator for padding
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # ensure the tokenizer is passed for proper data collation
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ],  # early stopping callback
        compute_metrics=compute_metrics,  # compute metrics function
    )

    show_modules(model)  # check the modules
    show_trainable(model)  # check trainable parameters
    confirm_learning(trainer, model)  # check the gradients

    # 6. Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")

    # 7. Save the Finetuned Model and Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
    # evaluate()
