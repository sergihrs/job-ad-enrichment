import pandas as pd
from src.models.stat_models.stat_model_salary import stat_model_salary
from src.models.stat_models.stat_model_work_arr import stat_model_work_arr
from src.models.stat_models.stat_model_seniority import stat_model_seniority
from src.models.fine_tuned_models.FacebookOpt350mModel import FacebookOpt350mModel
from src.config import MetaP


def _run_facebook_opt359m(data: dict[pd.DataFrame]) -> None:
  """
  Train the claude_3_5_haiku model.
  """
  for dataset_name, (train_data_name, test_data_name) in MetaP.DATASETS_FOR_FACEBOOK_OPT350M.items():
    train_data = data[train_data_name]
    test_data = data[test_data_name]
    
    model = FacebookOpt350mModel(dataset_name, train_data=train_data, val_data=test_data)
    model.setup_and_train()
    model.save_model()


def run_stat_models(data: dict[pd.DataFrame]) -> None:
  stat_model_seniority(seniority_dev=data['seniority_dev'])
  stat_model_work_arr(work_arr_dev=data['work_arr_dev'])
  stat_model_salary(salary_dev=data['salary_dev'])


def run_fine_tuned_models(data: dict[pd.DataFrame]) -> None:
  _run_facebook_opt359m(data)

if __name__ == '__main__':
  pass