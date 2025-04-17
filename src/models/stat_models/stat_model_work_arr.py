import pandas as pd
from src.models.stat_models.stat_model_functions import stat_model_classifier

def stat_model_work_arr(work_arr_dev: pd.DataFrame) -> None:
  stat_model_classifier(work_arr_dev, 'work_arr')

if __name__ == '__main__':
  pass