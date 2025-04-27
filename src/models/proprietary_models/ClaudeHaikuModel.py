from anthropic import Anthropic
import pandas as pd
from src.config import MetaP
import os
from sklearn.metrics import accuracy_score

class ClaudeHaikuModel:
  def __init__(
    self,
    dataset_name: str,
    client: Anthropic,
    prompt_start: str,
  ):
    self.name = 'claude'
    self.dataset_name = dataset_name
    self.prompt_start = prompt_start
    self.x_column_name = 'job_ad_details'
    self.y_column_name = 'y_true_grouped'
    self.client = client
    
    os.makedirs(os.path.join(MetaP.MODELS_DIR, self.name), exist_ok=True)
    
  @property
  def n_labels(self) -> int:
    return len(self.label_to_id)
  
  def _get_prompts(self, inputs: pd.DataFrame) -> None:
    job_ads = list(inputs[self.x_column_name])
    prompts = [self.prompt_start + ':\n' + job_ad + '\nAnswer:' for job_ad in job_ads]
    prompts = pd.Series(prompts)
    return prompts
    
  def _train(self):
    # No training is done here, as the model is already trained.
    pass
    
  def save_model(self):
    # N/A
    pass
    
  def setup_and_train(self):
    # Model already trained, so no setup or training is needed.
    pass
  
  def _get_predictions(self, response: pd.DataFrame) -> pd.DataFrame:
    # No post-processing is needed, as the model already returns the predictions.
    return response
    
  def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
    def query_claude(prompt):
      prompt = str(prompt)
      response = self.client.messages.create(
          model="claude-3-5-haiku-latest",
          max_tokens=10,
          messages=[{"role": "user", "content": prompt}]
      )
      return response.content[0].text

    prompt_data = self._get_prompts(inputs)
    response = prompt_data.apply(query_claude)
    predictions = self._get_predictions(response)
    labels = inputs[self.y_column_name].str.strip().str.lower()

    predictions_df = pd.DataFrame({
        "predictions": list(predictions),
        "labels": list(labels)
    })
    predictions_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_predictions.csv'), index=False)
    
    # Get accuracy by label
    predictions_df['predictions'] = predictions_df['predictions'].str.strip().str.lower()
    
    accuracy_per_label = predictions_df.groupby("labels").apply(lambda g: (g["predictions"] == g["labels"]).mean())
    accuracy_df = accuracy_per_label.reset_index()
    accuracy_df.columns = ["label", "accuracy"]
    accuracy_df.to_csv(os.path.join(MetaP.MODELS_DIR, self.name, f'{self.name}_{self.dataset_name}_val_accuracy.csv'), index=False)

