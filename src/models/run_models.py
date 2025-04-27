import pandas as pd
from src.models.stat_models.stat_model_salary import stat_model_salary
from src.models.stat_models.stat_model_work_arr import stat_model_work_arr
from src.models.stat_models.stat_model_seniority import stat_model_seniority
from src.models.fine_tuned_models.FacebookOpt350mModel import FacebookOpt350mModel
from src.models.fine_tuned_models.BERTModel import BERTModel
from src.models.proprietary_models.ClaudeHaikuModel import ClaudeHaikuModel
from src.config import MetaP, HyperP
from src.ancillary_functions import connect_to_anthropic


def get_bad_predictions(
  model_name: str,
  validation_data: pd.DataFrame,
  predictions: pd.DataFrame,
) -> None:
  """
  Extract the first few bad predictions from the validation data and save them to a CSV file.
  """
  bad_predictions = validation_data[validation_data['y_true_grouped'] != predictions['y_pred']].head(5)
  bad_predictions.to_csv(f'bad_predictions_{model_name}.csv', index=False)

def run_claude_haiku(data: dict[pd.DataFrame]) -> None:
  print('Running Claude Haiku model...')
      
  try:
    client = connect_to_anthropic()
    for dataset_name, (test_data_name, prompt_start) in HyperP.DATASETS_AND_PROMPTS_FOR_CLAUDE_MODELS.items():
      print(f'Running Claude Haiku model for {dataset_name}...')
      
      # Only testing data for this Claude model
      test_data = data[test_data_name]
      
      model = ClaudeHaikuModel(dataset_name, client, prompt_start)
      model.setup_and_train()
      model.save_model()
      model.predict(test_data.head(5))
      
  except Exception as e:
    print(f"Error connecting to Anthropic API: {e}")
    return


def run_bert(data: dict[pd.DataFrame]) -> None:
  """
  Train the BERT model on the given datasets.
  """
  print('Running BERT model...')
  
  for dataset_name, (train_data_name, test_data_name) in HyperP.DATASETS_FOR_FINE_TUNED_MODELS.items():
    train_data = data[train_data_name]
    test_data = data[test_data_name]
    
    model = BERTModel(dataset_name, train_data=train_data, val_data=test_data)
    model.setup_and_train()
    model.save_model()
    model.predict()
  
  


def run_facebook_opt359m(
  data: dict[pd.DataFrame],
  test_data: pd.DataFrame=None,
  model_name: str=None
) -> None:
  """
  Train the Facebook Opt 350m model on the given datasets.
  """  
  print('Running Facebook Opt 350m model...')
  for dataset_name, (train_data_name, test_data_name) in HyperP.DATASETS_FOR_FINE_TUNED_MODELS.items():
    if model_name is not None and dataset_name != model_name:
      continue
    
    train_data = data[train_data_name]
    val_data = data[test_data_name]
    
    model = FacebookOpt350mModel(dataset_name, train_data=train_data, val_data=val_data)
    model.setup_and_train()
    model.save_model()
    model.predict()
  
    if test_data is not None:
      return model.predict(test_data)


def run_stat_models(data: dict[pd.DataFrame]) -> None:
  stat_model_seniority(seniority_dev=data['seniority_dev'])
  stat_model_work_arr(work_arr_dev=data['work_arr_dev'])
  stat_model_salary(salary_dev=data['salary_dev'])


def run_fine_tuned_models(data: dict[pd.DataFrame]) -> None:
  run_bert(data)
  # run_facebook_opt359m(data)
  
def run_proprietary_models(data: dict[pd.DataFrame]) -> None:
  # run_claude_haiku(data)
  pass

if __name__ == '__main__':
  pass