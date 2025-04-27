import os
import argparse
from src.config import MetaP, CMDArgs
import pandas as pd
import numpy as np

print('Importing Anthropic. Could take some time...')
from anthropic import Anthropic
import getpass

def get_bad_predictions(
  model_name: str,
  dataset_name: str,
  predictions_df: pd.DataFrame,
) -> None:
  """
  Extract the first few bad predictions from the validation data and save them to a CSV file.
  """  
  mask = np.array(predictions_df['labels']) != np.array(predictions_df['predictions'])
  mismatches = predictions_df[mask]
  mismatches['error_type'] = list(zip(mismatches['labels'], mismatches['predictions']))
  one_example_per_error_type = mismatches.drop_duplicates(subset='error_type')
  one_example_per_error = one_example_per_error_type.drop(columns=['error_type'])
  one_example_per_error_type.to_csv(os.path.join(MetaP.MODELS_DIR, model_name, f'{dataset_name}_bad_predictions.csv'), index=False)

def verify_file(file_path) -> None:
  """
  Verify if the file exists and is a CSV file.
  """
  if CMDArgs.FILE is None:
    return
  
  if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File {file_path} does not exist.")
  
  if not file_path.endswith('.csv'):
    raise ValueError(f"File {file_path} is not a CSV file.")
  
  # Try reading the first two lines of the file to check if it's a valid CSV
  with open(file_path, 'r', encoding='utf-8') as file:
    # Read the first line
    header = file.readline()
    if not header.strip():
      raise ValueError(f"File {file_path} is empty or does not contain valid CSV data.")
    
    # Read the second line to check for data
    line = file.readline()
    if not line.strip():
      raise ValueError(f"File {file_path} is empty or does not contain valid CSV data.")
    
    if 'job_ad_details' not in header:
      raise ValueError(f"File {file_path} does not contain the required 'job_ad_details' column.")
    
    if 'y_true' not in header:
      raise ValueError(f"File {file_path} does not contain the required 'y_true' column.")
  
def custom_round(x):
  import math
  frac = x - int(x)
  if frac <= 0.5:
    return math.floor(x)
  else:
    return math.ceil(x)
  

def save_user_testing_data(data: pd.DataFrame) -> None:
  """
  Save the user testing data to a CSV file.
  """
  # Strip the input file name of the extension
  input_file = CMDArgs.FILE
  
  if input_file.endswith('.csv'):
    input_file = input_file[:-4]
    
  data.to_csv(os.path.join(MetaP.MODELS_USER_TESTING_DIR, f'{input_file}_preds.csv'), index=False)  
  print(f"User testing data saved to {MetaP.MODELS_USER_TESTING_DIR}/{input_file}_preds.csv")
  
def _create_dirs():
  # Directories
  folders = [
    MetaP.DATA_DIR,
    MetaP.REPORT_DIR,
    MetaP.STAT_MODELS_DIR,
    MetaP.MODELS_DIR,
    MetaP.MODELS_USER_TESTING_DIR,
  ]

  for folder in folders:
    os.makedirs(folder, exist_ok=True)
    
def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', required=False, type=str, help='File must be a string.')
  parser.add_argument('--target', required=False, choices=['seniority', 'work_arr', 'salary'], help='Target must be one of seniority or work_arr.')
  parser.add_argument('--stat', required=False, action='store_true', help='Run the statistical model.')

  args = parser.parse_args()
  
  try:
    CMDArgs.parse_args(args)
    verify_file(CMDArgs.FILE)
  except Exception as e:
    print(f"Error parsing arguments: {e}")
    exit(1)
    
    
def connect_to_anthropic() -> str:
  """
  Get the Anthropic API key from the environment variable.
  """
  print("Connecting to Anthropic API...")
  api_key = os.getenv('ANTHROPIC_API_KEY')
  api_key = "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"
  if not api_key:
    api_key = getpass.getpass('Enter your Anthropic API key: ')
  
  # Attempt to connect to the Anthropic API by making a simple API call
  client = Anthropic(api_key=api_key)
  response = client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=1,
      messages=[
          {"role": "user", "content": "Hello?"}
      ]
  )
  
  return client
    
def setup():
  print('Setting up directories and parsing arguments...')
  _create_dirs()
  _parse_args()

if __name__ == '__main__':
  pass
