import os
import pandas as pd
from src.config import MetaP
from src.DataPreProcessor import DataPreProcessor
from src.config import HyperP
from datasets import Dataset



def _preprocess_seniority(
  seniority_dev: pd.DataFrame,
  seniority_test: pd.DataFrame,
  save_to_data_dir: bool=True
):
    """
    Preprocesses the seniority dataset and return train and test datasets
    """

    # Remove html tags from job_ad_details
    seniority_dev["job_ad_details"] = seniority_dev["job_ad_details"].str.replace(
        r"<[^>]+>", " ", regex=True
    )

    # Remove &nbsp; and \n from job_ad_details
    seniority_dev["job_ad_details"] = seniority_dev["job_ad_details"].str.replace(
        "&[a-zA-Z0-9]+;", " ", regex=True
    )

    # Merge multiple spaces into one
    seniority_dev["job_ad_details"] = seniority_dev["job_ad_details"].str.replace(
        r"\s+", " ", regex=True
    )

    # Merge columns that mean the same
    seniority_dev["y_true"].replace(
        {
            "entry-level": "entry level",
            "mid-senior": "intermediate",
            "mid-level": "intermediate",
            "board": "director",
        },
        inplace=True,
    )

    # Filter columns with less than 8 occurrences
    seniority_dev["value_count"] = seniority_dev["y_true"].map(seniority_dev["y_true"].value_counts())
    seniority_dev = seniority_dev[seniority_dev["value_count"] >= 8]

    # Merge all text inputs into one column
    merge_text = lambda x: (
        x["job_title"]
        + ". "
        + x["job_summary"]
        + ". "
        + x["job_ad_details"]
        + ". "
        + x["classification_name"]
        + ". "
        + x["subclassification_name"]
    )
    seniority_dev["job_text"] = seniority_dev.apply(merge_text, axis=1)
    seniority_test["job_text"] = seniority_test.apply(merge_text, axis=1)

    # Save to csv files. Only text and y_true
    seniority_dev = seniority_dev[["job_text", "y_true"]]
    seniority_test = seniority_test[["job_text", "y_true"]]
    # seniority_dev.rename(columns={"y_true": "labels"}, inplace=True)
    # seniority_test.rename(columns={"y_true": "labels"}, inplace=True)

    # Add a new column "labels" with the integer values of the unique labels (use both train and test)
    unique_labels = set(seniority_dev["y_true"].unique()) | set(seniority_test["y_true"].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    seniority_dev["labels"] = seniority_dev["y_true"].map(label_to_id)
    seniority_test["labels"] = seniority_test["y_true"].map(label_to_id)
    
    if save_to_data_dir:
      seniority_dev.to_csv(os.path.join(MetaP.DATA_DIR, "seniority_train.csv"), index=False)
      seniority_test.to_csv(os.path.join(MetaP.DATA_DIR, "seniority_test.csv"), index=False)
         
    seniority_dev = Dataset.from_pandas(seniority_dev)
    seniority_test = Dataset.from_pandas(seniority_test)
  
    return seniority_dev, seniority_test, id_to_label, label_to_id


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
  
  seniority_dev_for_multiclass, seniority_test_for_multiclass, \
    id_to_label, label_to_id = _preprocess_seniority(seniority_dev, seniority_test)

  seniority_dev = _merge_seniority_data(seniority_dev)

  all_data = {
    'salary_dev': salary_dev,
    'salary_test': salary_test,
    'seniority_dev': seniority_dev,
    'seniority_test': seniority_test,
    'work_arr_dev': work_arr_dev,
    'work_arr_test': work_arr_test,
    'unlabelled_dev': unlabelled_dev,
    'seniority_dev_for_multiclass': seniority_dev_for_multiclass,
    'seniority_test_for_multiclass': seniority_test_for_multiclass,
    'id_to_label': id_to_label,
    'label_to_id': label_to_id,
  }

  return all_data

def preprocess_data(data: dict[pd.DataFrame]) -> dict[pd.DataFrame]:
  """
  Preprocesses the data by removing duplicates and NaN values.
  data: dict: The data to preprocess.
  Returns:
    data: dict: The preprocessed data.
  """
  preprocessor = DataPreProcessor(data)
  data = preprocessor.clean_data()
  preprocessor.save_log()
  return data



if __name__ == "__main__":
  pass