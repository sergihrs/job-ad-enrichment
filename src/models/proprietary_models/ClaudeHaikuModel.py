from anthropic import Anthropic
import pandas as pd
from src.config import MetaP
import os

class ClaudeHaikuModel:
  def __init__(
    self,
    dataset_name: str,
    client: Anthropic,
  ):
    self.name = 'claude'
    self.dataset_name = dataset_name
    self.x_column_name = 'job_ad_details'
    self.y_column_name = 'y_true_grouped'
    
    self._get_prompts()
    self.client = client
    
  @property
  def n_labels(self) -> int:
    return len(self.label_to_id)
  
  def _get_prompts(self, input: pd.DataFrame) -> None:
    job_ads = list(input[self.x_column_name])
    prompts = ["Classify the following text as either 'remote', 'onsite' or 'hybrid'. Respond with one word only:\n" + job_ad + "'\nAnswer:" for job_ad in job_ads]
    prompts_df = pd.DataFrame({'prompt': prompts})
    return prompts_df
    
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
    
  def predict(self, input: pd.DataFrame) -> pd.DataFrame:
    def query_claude(prompt):
      response = self.client.messages.create(
          model="claude-3-5-haiku-latest",  # change model as needed
          max_tokens=10,
          messages=[{"role": "user", "content": prompt}]
      )
      return response.content[0].text

    prompt_data = self._get_prompts(input)
    response = prompt_data.apply(query_claude)
    predictions = self._get_predictions(response)
    # Apply the trained model on val_data
    predictions = self.trainer.predict(self.val_data)
    labels = self.data[self.y_column_name]

    predictions_df = pd.DataFrame({
        "predictions": predictions,
        "labels": labels
    })
    predictions_df.to_csv(os.path.join(MetaP.MODELS_DIR, f'{self.name}_{self.dataset_name}_val_predictions.csv'), index=False)

