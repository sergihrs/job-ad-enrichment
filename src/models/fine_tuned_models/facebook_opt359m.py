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

class FacebookOpt359mModel:
  def __init__(
    self,
    dataset_name: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
  ):
    self.dataset_name = dataset_name
    self.model_name = 'facebook/opt-350m'
    
    self._get_prompt_start_and_end()
    self._get_data(train_data, val_data)
    
    self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_labels)
    
  @property
  def n_labels(self) -> int:
    return len(self.label_to_id)
  
  def _get_prompt_start_and_end(self) -> None:
    text = HyperP.FACEBOOK_OPT350M_PROMPTS[self.dataset_name]
    self.prompt_start = f'What is the {text.lower()} of this Job?\Job: '
    self.prompt_end = f'\{text}: '
    
  def _get_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
    # Get only the columns needed for training the facebook_opt359m model
    x_column_name = 'job_ad_details'
    y_column_name = 'y_true_grouped'
    
    assert x_column_name in train_data.columns, f"Column {x_column_name} not found in train data."
    assert y_column_name in train_data.columns, f"Column {y_column_name} not found in train data."
    
    train_data = train_data[[x_column_name, y_column_name]].copy()
    val_data = val_data[[x_column_name, y_column_name]].copy()
    
    # Convert the target to integers
    unique_labels = set(train_data[y_column_name].unique()) | set(val_data[y_column_name].unique())
    self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
    self.id_to_label = {i: label for label, i in self.label_to_id.items()}
    train_data['label'] = train_data[y_column_name].map(self.label_to_id)
    val_data['label'] = val_data[y_column_name].map(self.label_to_id)
    
    self.train_data = Dataset.from_pandas(train_data)
    self.val_data = Dataset.from_pandas(val_data)
    
  def _tokenise(self):
    self.train_data = self.dataset.train_data.map(lambda samples: self.tokeniser(samples["job_text"]), batched=True)
    self.val_data = self.dataset.val_data.map(lambda samples: self.tokeniser(samples["job_text"]), batched=True)
    
  def _set_peft_model(self):
    self.peft_config = LoraConfig(
      task_type=TaskType.SEQ_CLS, # LoRA for sequence classification
      inference_mode=False,
      r=8,
      lora_alpha=32,
      lora_dropout=0.1
    )
    
    self.model = get_peft_model(self.model, self.peft_config)
    
  def _preprocess_inputs(self, input: Dataset) -> dict:
    def preprocess(input: Dataset) -> dict:
      prompt = f'{self.prompt_start}{input["job_ad_details"]}{{self.prompt_end}}'

      tokenised = self.tokeniser(prompt, padding='max_length', truncation=True, max_length=128)
      tokenised['label'] = input['label']
      return tokenised
    
    self.train_data = self.train_data.map(preprocess, remove_columns=self.train_data.column_names)
    test_data = self.test_data.map(preprocess, remove_columns=self.test_data.column_names)
    
  def _train(self):
    def compute_metrics(eval_pred):
      logits, labels = eval_pred
      preds = logits.argmax(-1)
      return {"accuracy": accuracy_score(labels, preds)}

    self.trainer = Trainer(
      model=self.model,
      args=TrainingArguments(
        output_dir=os.path.join(MetaP.MODELS_DIR, self.model_name),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=500,
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
    self.model.save_pretrained(os.path.join(MetaP.MODELS_DIR, self.model_name))
    self.tokeniser.save_pretrained(os.path.join(MetaP.MODELS_DIR, self.model_name))
    
  def setup_and_train(self):
    self._tokenise()
    self._set_peft_model()
    self._train()
    
  def predict(self, input: pd.DataFrame) -> pd.DataFrame:
    # Apply the trained model on val_data
    predictions = self.trainer.predict(self.val_data)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    predictions_df = pd.DataFrame({
        "predictions": preds,
        "labels": labels
    })
    predictions_df.to_csv(os.path.join(MetaP.MODELS_DIR, f'{self.dataset_name}_val_predictions.csv'), index=False)

def _get_models(data: dict[pd.DataFrame]) -> dict[FacebookOpt359mModel]:
  """
  Get the models for the facebook_opt359m model.
  :param datasets: Dictionary of DataFrames with prepared data.
  :return: Dictionary of models.
  """
  models = {}
  for dataset_name, (train_data_name, test_data_name) in MetaP.DATASETS_FOR_FACEBOOK_OPT350M.items():
    train_data = data[train_data_name]
    test_data = data[test_data_name]
    
    models[dataset_name] = FacebookOpt359mModel(dataset_name, train_data=train_data, val_data=test_data)
  
  return models


def run_facebook_opt359m(data: dict[pd.DataFrame]) -> None:
  """
  Train the facebook_opt359m model.
  """
  models = _get_models(data)
  
  for model in models.values():
    model.setup_and_train()
    model.save_model()
  

if __name__ == '__main__':
  pass