import pandas as pd
from src.models.stat_models.stat_model_functions import stat_model_classifier

def stat_model_work_arr(work_arr_dev: pd.DataFrame) -> None:
  stat_model_classifier(work_arr_dev, 'work_arr')
  
def stat_model_work_arr_on_test_data(work_arr_dev: pd.DataFrame, test_data: pd.DataFrame) -> None:
  """
  Train on the training data then predict on the test data.
  Args:
    work_arr_dev (pd.DataFrame): The work_arr development data.
    test_data (pd.DataFrame): The test data to predict on.
  """
  model = stat_model_classifier(work_arr_dev, 'work_arr')

  # Predict on validation set and report accuracy
  val_predictions = model.predict(test_data['job_ad_details'])
  test_data['predictions'] = val_predictions
  test_data['correct'] = test_data['y_true'] == test_data['predictions']
  
  return test_data

if __name__ == '__main__':
  pass