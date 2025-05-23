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

class FacebookOpt350mModel:
  def __init__(
    self,
    dataset_name: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
  ):
    self.name = 'facebook'
    self.dataset_name = dataset_name
    self.model_name = 'facebook/opt-350m'
    self.x_column_name = 'consolidated_fields'
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
    text = HyperP.FACEBOOK_OPT350M_PROMPTS[self.dataset_name]
    self.prompt_start = f'What is the {text.lower()} of this Job?\Job: '
    self.prompt_end = f'\{text}: '
    
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
    self.val_data_x_column = self.val_data[self.x_column_name].copy()
    self.train_data = self.train_data.map(lambda samples: self.tokeniser(samples[self.x_column_name]), batched=True)
    self.val_data = self.val_data.map(lambda samples: self.tokeniser(samples[self.x_column_name]), batched=True)
    
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
    
  def predict(
    self,
    test_data: pd.DataFrame=None,
  ) -> pd.DataFrame:
    if test_data is None:
      print('Running prediction on validation data...')
      data_for_prediction = self.val_data
    else:
      print('Running prediction on test data...')
      data_for_prediction = test_data[[self.x_column_name, self.y_column_name]].copy()
      data_for_prediction['label'] = data_for_prediction[self.y_column_name].map(self.label_to_id)
      data_for_prediction = Dataset.from_pandas(data_for_prediction)
      
      self.val_data_x_column = test_data[self.x_column_name].copy()
      data_for_prediction = data_for_prediction.map(lambda samples: self.tokeniser(samples[self.x_column_name]), batched=True)
      
      def preprocess(input: Dataset) -> dict:
        prompt = f'{self.prompt_start}{input[self.x_column_name]}{self.prompt_end}'

        tokenised = self.tokeniser(prompt, padding='max_length', truncation=True, max_length=128)
        tokenised['label'] = input['label']
        return tokenised
      
      data_for_prediction = data_for_prediction.map(preprocess, remove_columns=data_for_prediction.column_names)
      
    # Apply the trained model on val_data
    predictions = self.trainer.predict(data_for_prediction)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    predictions_df = pd.DataFrame({
        "x": self.val_data_x_column,
        "predictions": preds,
        "labels": labels
    })
    
    # Convert the ids back to labels
    predictions_df["predictions"] = predictions_df["predictions"].map(self.id_to_label)
    predictions_df["labels"] = predictions_df["labels"].map(self.id_to_label)
    
    predictions_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_predictions.csv'), index=False)
    
    accuracy_per_label = predictions_df.groupby("labels").apply(lambda g: (g["predictions"] == g["labels"]).mean())
    accuracy_df = accuracy_per_label.reset_index()
    accuracy_df.columns = ["label", "accuracy"]
    accuracy_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_accuracy.csv'), index=False)
    
    get_bad_predictions(
      model_name=self.name,
      dataset_name=self.dataset_name,
      predictions_df=predictions_df
    )
    
    return predictions_df
    
    

