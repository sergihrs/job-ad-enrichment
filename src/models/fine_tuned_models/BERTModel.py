from datasets import Dataset
from src.config import MetaP, HyperP
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import os
from src.ancillary_functions import get_bad_predictions

class BERTModel:
  def __init__(
    self,
    dataset_name: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
  ):
    self.name = 'bert'
    self.dataset_name = dataset_name
    self.model_name = 'bert-base-uncased'
    self.x_column_name = 'job_ad_details'
    self.y_column_name = 'y_true_grouped'
    
    os.makedirs(os.path.join(MetaP.MODELS_DIR, self.name), exist_ok=True)
    
    self._get_prompt_start_and_end()
    self._get_data(train_data, val_data)
    
    self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_labels)
    
  @property
  def n_labels(self) -> int:
    return len(self.label_to_id)
  
  def _get_prompt_start_and_end(self) -> None:
    text = HyperP.BERT_PROMPTS[self.dataset_name]
    self.prompt_start = text
    self.prompt_end = ''
    
  def _get_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:    
    train_data = train_data[[self.x_column_name, self.y_column_name]].copy()
    val_data = val_data[[self.x_column_name, self.y_column_name]].copy()
    
    # Convert the target to integers
    unique_labels = set(train_data[self.y_column_name].unique()) | set(val_data[self.y_column_name].unique())
    self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
    self.id_to_label = {i: label for label, i in self.label_to_id.items()}
    train_data['label'] = train_data[self.y_column_name].map(self.label_to_id)
    val_data['label'] = val_data[self.y_column_name].map(self.label_to_id)
    
    self.train_data = Dataset.from_pandas(train_data)
    self.val_data = Dataset.from_pandas(val_data)
    
  def _tokenise(self):
    self.val_data_y_column = self.val_data[self.y_column_name].copy()
    self.train_data = self.train_data.map(lambda samples: self.tokeniser(samples[self.x_column_name], padding='max_length', truncation=True), batched=True)
    self.val_data = self.val_data.map(lambda samples: self.tokeniser(samples[self.x_column_name], padding='max_length', truncation=True), batched=True)
    
  def _set_peft_model(self):
    self.peft_config = LoraConfig(
      task_type=TaskType.SEQ_CLS, # LoRA for sequence classification
      inference_mode=False,
      r=8,
      lora_alpha=32,
      lora_dropout=0.1
    )
    
    self.model = get_peft_model(self.model, self.peft_config)
    
  def _preprocess_inputs(self) -> dict:
    def preprocess(input: Dataset) -> dict:
      prompt = f'{self.prompt_start}{input[self.x_column_name]}{self.prompt_end}'

      tokenised = self.tokeniser(prompt, padding='max_length', truncation=True, max_length=128)
      tokenised['label'] = input['label']
      return tokenised
    
    self.train_data = self.train_data.map(preprocess, remove_columns=self.train_data.column_names)
    self.val_data = self.val_data.map(preprocess, remove_columns=self.val_data.column_names)
    
  def _train(self):
    def compute_metrics(eval_pred):
      logits, labels = eval_pred
      preds = logits.argmax(-1)
      return {"accuracy": accuracy_score(labels, preds)}

    self.trainer = Trainer(
      model=self.model,
      args=TrainingArguments(
        output_dir=os.path.join(MetaP.MODELS_DIR, self.name),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=HyperP.MAX_TRAINING_STEPS,
        warmup_steps=5,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy='steps',
        eval_steps=10,
        save_strategy='no',
      ),
      train_dataset=self.train_data,
      eval_dataset=self.val_data,
      tokenizer=self.tokeniser,
      compute_metrics=compute_metrics,
      data_collator=DataCollatorWithPadding(self.tokeniser),
    )

    self.model.config.use_cache = False
    self.trainer.train()
    
  def save_model(self):
    self.model.save_pretrained(os.path.join(MetaP.MODELS_DIR, self.name))
    self.tokeniser.save_pretrained(os.path.join(MetaP.MODELS_DIR, self.name))
    
  def setup_and_train(self):
    self._tokenise()
    self._set_peft_model()
    self._preprocess_inputs()
    self._train()
    
  def predict(self) -> pd.DataFrame:
    # Apply the trained model on val_data
    predictions = self.trainer.predict(self.val_data)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    predictions_df = pd.DataFrame({
        "predictions": preds,
        "labels": labels
    })
    predictions_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_predictions.csv'), index=False)
    
    accuracy_per_label = predictions_df.groupby("labels").apply(lambda g: (g["predictions"] == g["labels"]).mean())
    accuracy_df = accuracy_per_label.reset_index()
    accuracy_df.columns = ["label", "accuracy"]
    accuracy_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_accuracy.csv'), index=False)
    
    get_bad_predictions(
      model_name=self.name,
      x_field=self.val_data_y_column,
      predictions_df=predictions_df
    )
    

