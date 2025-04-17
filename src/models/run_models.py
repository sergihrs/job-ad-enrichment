import pandas as pd
from src.models.stat_models.stat_model_salary import stat_model_salary
from src.models.stat_models.stat_model_work_arr import stat_model_work_arr
from src.models.stat_models.stat_model_seniority import stat_model_seniority

def run_stat_models(data: dict[pd.DataFrame]) -> None:
  stat_model_seniority(seniority_dev=data['seniority_dev'])
  stat_model_work_arr(work_arr_dev=data['work_arr_dev'])
  stat_model_salary(salary_dev=data['salary_dev'])

if __name__ == '__main__':
  pass