import os
import argparse
from src.config import MetaP, CMDArgs

print('Importing Anthropic. Could take some time...')
from anthropic import Anthropic

def verify_file(file_path) -> None:
  """
  Verify if the file exists and is a CSV file.
  """
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
  
def custom_round(x):
  import math
  frac = x - int(x)
  if frac <= 0.5:
    return math.floor(x)
  else:
    return math.ceil(x)
  
  
def _create_dirs():
  # Directories
  folders = [
    MetaP.DATA_DIR,
    MetaP.REPORT_DIR,
    MetaP.STAT_MODELS_DIR,
    MetaP.MODELS_DIR,
  ]

  for folder in folders:
    os.makedirs(folder, exist_ok=True)
    
def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--target', required=False, choices=['seniority', 'work_arr'], help='Target must be one of seniority or work_arr.')
  parser.add_argument('--file', required=False, type=str, help='File must be a string.')
  parser.add_argument('--stat', required=False, action='store_true', help='Run the statistical model.')

  args = parser.parse_args()
  
  try:
    CMDArgs.parse_args(args)
    verify_file(CMDArgs.FILE)
  except Exception as e:
    print(f"Error parsing arguments: {e}")
    
    
def connect_to_anthropic() -> str:
  """
  Get the Anthropic API key from the environment variable.
  """
  print("Connecting to Anthropic API...")
  api_key = os.getenv('ANTHROPIC_API_KEY')
  if not api_key:
    api_key = input("Enter your Anthropic API key: ")
  
  # Attempt to connect to the Anthropic API by making a simple API call
  client = Anthropic(api_key=api_key)
  response = client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=1,
      messages=[
          {"role": "user", "content": "Hello?"}
      ]
  )

  
  print("Connected to Anthropic API.")
  
  return client
    
def setup():
  _create_dirs()
  
  if MetaP.DO_PARSE_ARGS:
    _parse_args()

if __name__ == '__main__':
  pass
