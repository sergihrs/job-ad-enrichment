import os
import pandas as pd
from src.config import MetaP
from src.data.PreProcessor import PreProcessor
from src.config import HyperP
from datasets import Dataset


def _read_data_file(file_path: str, file_name: str, col_names: list[str]=None) -> pd.DataFrame:
  """
  Reads a CSV file from the specified path and returns it as a pandas DataFrame.
  file_path: str: The path to the data file.
  file_name: str: The name of the data file.
  col_names: list[str]: The column names to use for the DataFrame.
  If None, the column names will be inferred from the file.
  Returns:
    df: pd.DataFrame: The data as a pandas DataFrame.
  Raises:
    FileNotFoundError: If the file does not exist.
  """
  data_file_name = os.path.join(file_path, file_name)
  if not os.path.exists(data_file_name):
    raise FileNotFoundError(f"The file {data_file_name} does not exist.")
  df = pd.read_csv(data_file_name)

  if col_names is not None:
    df.columns = col_names

  return df

def _split_salary_y_true(salary_data: pd.DataFrame) -> pd.DataFrame:
  """
  Splits the salary data into min, max, currency, and frequency columns.
  salary_dev_y_true: pd.DataFrame: The salary data to split.
  Returns:
    salary_dev_y_true: pd.DataFrame: The split salary data.
  """
  salary_data[['y_true_salary_min', 'y_true_salary_max', 'y_true_currency', 'y_true_frequency']] = \
    salary_data['y_true'].str.split(HyperP.SALARY_DELIMITER, expand=True)
  
  salary_data['y_true_salary_min'] = salary_data['y_true_salary_min'].astype(int)
  salary_data['y_true_salary_max'] = salary_data['y_true_salary_max'].astype(int)
  salary_data['y_true_currency'] = salary_data['y_true_currency'].astype(str).apply(lambda x: x.strip())
  salary_data['y_true_frequency'] = salary_data['y_true_frequency'].astype(str).apply(lambda x: x.strip())

  return salary_data


def _merge_seniority_data(seniority_dev: pd.DataFrame) -> pd.DataFrame:
  seniority_dev['y_true_merged'] = seniority_dev['y_true'].replace(to_replace=HyperP.SENIORITY_REPLACE)

  # Exclude items in seniority_dev with fewer than 8 counts
  seniority_dev = seniority_dev[seniority_dev['y_true_merged'].map(seniority_dev['y_true_merged'].value_counts()) >= HyperP.SENIORITY_MIN_COUNT]

  HyperP.SENIORITY_KEYWORDS = seniority_dev['y_true_merged'].unique()

  return seniority_dev


def load_data(file_path: str=f'./{MetaP.DATA_DIR}/') -> dict[object]:
  """
  Reads the data files from the specified path and returns them as a dictionary.
  file_path: str: The path to the data files.
  Returns:
    all_data: dict: A dictionary containing the dataframes or HuggingFace Datasets.
  """
  salary_dev = _read_data_file(file_path=file_path, file_name='salary_labelled_development_set.csv')
  salary_test = _read_data_file(file_path=file_path, file_name='salary_labelled_test_set.csv')
  seniority_dev = _read_data_file(file_path=file_path, file_name='seniority_labelled_development_set.csv')
  seniority_test = _read_data_file(file_path=file_path, file_name='seniority_labelled_test_set.csv')
  work_arr_dev = _read_data_file(file_path=file_path, file_name='work_arrangements_development_set.csv',
                                 col_names=['job_id', 'job_ad_details', 'y_true'])
  work_arr_test = _read_data_file(file_path=file_path, file_name='work_arrangements_test_set.csv',
                                  col_names=['job_id', 'job_ad_details', 'y_true'])
  unlabelled_dev = _read_data_file(file_path=file_path, file_name='unlabelled_development_set.csv')

  # For salary dev and test only, split the y_true field into min, max, currency, and frequency
  salary_dev = _split_salary_y_true(salary_dev)
  salary_test = _split_salary_y_true(salary_test)

  all_data = {
    'salary_dev': salary_dev,
    'salary_test': salary_test,
    'seniority_dev': seniority_dev,
    'seniority_test': seniority_test,
    'work_arr_dev': work_arr_dev,
    'work_arr_test': work_arr_test,
    'unlabelled_dev': unlabelled_dev,
  }

  return all_data

def preprocess_data(data: dict[pd.DataFrame]) -> dict[pd.DataFrame]:
  """
  Preprocesses the data by removing duplicates and NaN values.
  data: dict: The data to preprocess.
  Returns:
    data: dict: The preprocessed data.
  """
  preprocessor = PreProcessor(data)
  data = preprocessor.clean_data()
  preprocessor.save_log()
  return data



if __name__ == "__main__":
  pass