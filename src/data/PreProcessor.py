import pandas as pd
from src.config import MetaP, HyperP
from src.models.stat_models.stat_model_functions import clean_html

class PreProcessor:
  def __init__(self, data):
    self.data = data
    self.log = pd.DataFrame(columns=['dataset_name', 'message', 'detail', 'value'])

  def _log(self, dataset_name: str, message: str, detail: str, value: float) -> None:
    """
    Log messages to the log DataFrame.
    :param dataset_name: Name of the dataset.
    :param message: Log message.
    :param detail: Detailed message.
    :param value: Value associated with the log message.
    """
    self.log = pd.concat([
      self.log,
      pd.DataFrame({
        'dataset_name': [dataset_name],
        'message': [message],
        'detail': [detail],
        'value': [value]
      })
    ])

  def _remove_duplicate_values_from_field(self, dataset_name: str, field_name: str):
    """
    Remove duplicate values from a specific field in the dataset.
    :param dataset_name: Name of the dataset.
    :param field_name: Name of the field to remove duplicates from.
    """
    if dataset_name in self.data:
      if field_name in self.data[dataset_name].columns:
        self._log(dataset_name, f'{field_name}_duplicates_removal', 'row_count_before', self.data[dataset_name].shape[0])
        self._log(dataset_name, f'{field_name}_duplicates_removal', 'n_unique_before', self.data[dataset_name][field_name].nunique())
        self.data[dataset_name] = self.data[dataset_name].drop_duplicates(subset=[field_name])
        self._log(dataset_name, f'{field_name}_duplicates_removal', 'row_count_after', self.data[dataset_name].shape[0])
        self._log(dataset_name, f'{field_name}_duplicates_removal', 'n_unique_after', self.data[dataset_name][field_name].nunique())
      else:
        raise ValueError(f"Field {field_name} not found in dataset {dataset_name}.")
    else:
      raise ValueError(f"Dataset {dataset_name} not found in data.")
    
  def _remove_duplicates(self):
    """
    Remove duplicate job IDs from the dataset.
    :param data: Dictionary of DataFrames.
    :return: DataFrame with duplicates removed.
    """
    for dataset_name, dataset in self.data.items():
      if isinstance(dataset, pd.DataFrame):
        if 'job_id' in dataset.columns:
          self._remove_duplicate_values_from_field(dataset_name, 'job_id')
        
        if 'job_ad_details' in dataset.columns:
          self._remove_duplicate_values_from_field(dataset_name, 'job_ad_details')
          
          
  def _consolidate_fields(
    self,
    field_names_to_consolidate: list[str]=HyperP.FIELD_NAMES_TO_CONSOLIDATE,
  ):
    for dataset_name in self.data:        
      for field_name in field_names_to_consolidate:
        if field_name in self.data[dataset_name].columns:
          self.data[dataset_name][field_name] = clean_html(self.data[dataset_name][field_name])
      
      field_names_to_consolidate_in_data = [
        field_name for field_name in field_names_to_consolidate if field_name in self.data[dataset_name].columns
      ]
      # Concatenate the fields into a new column
      self.data[dataset_name]['consolidated_fields'] = self.data[dataset_name][field_names_to_consolidate_in_data].agg(' '.join, axis=1)
      
  def _group_y_true(self):
    with pd.ExcelFile(MetaP.Y_TRUE_GROUPING_FILENAME) as xls:
      y_true_grouping = pd.read_excel(xls, sheet_name=None)
      
    for dataset_name in self.data:
      if 'y_true' in self.data[dataset_name].columns:
        if dataset_name.endswith(('_dev', '_test')):
          if dataset_name.endswith('_dev'):
            dataset_basename = dataset_name[:-4]
          else:
            dataset_basename = dataset_name[:-5]
          
          if dataset_basename in y_true_grouping:
            # Apply the grouping
            self.data[dataset_name] = self.data[dataset_name].merge(y_true_grouping[dataset_basename], on='y_true', how='left')
          else:
            self.data[dataset_name]['y_true_grouped'] = self.data[dataset_name]['y_true']
        else:
          self.data[dataset_name]['y_true_grouped'] = self.data[dataset_name]['y_true']

  def save_log(self):
    self.log.to_csv(f'{MetaP.REPORT_DIR}/preprocessor_log.csv', index=False)
    
  def clean_data(self):
    self._remove_duplicates()
    self._consolidate_fields()
    self._group_y_true()
    
    return self.data