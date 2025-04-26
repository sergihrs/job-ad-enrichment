import pandas as pd
import re
from matplotlib import pyplot as plt
from src.config import MetaP

def _summarise_salary_data(salary_dev: pd.DataFrame) -> None:
  """
  Summarise the salary data.

  Args:
    salary_dev (pd.DataFrame): The salary data DataFrame.

  Returns:
    None
  """
  # Plot a histogram of minimmum and maximum salary (different legends) excluding zeros

  # Annual
  annual_data = salary_dev[salary_dev['y_true_frequency'] == 'ANNUAL']
  min_data = annual_data['y_true_salary_min'].dropna()
  max_data = annual_data['y_true_salary_max'].dropna()
  min_data = min_data[(min_data > 0)]
  max_data = max_data[(max_data > 0)]

  plt.clf()
  plt.hist(min_data, bins=20, alpha=0.5, label='Minimum Salary')
  plt.hist(max_data, bins=20, alpha=0.5, label='Maximum Salary')
  plt.xlabel('Salary')
  plt.ylabel('Frequency')
  plt.title('Salary Distribution - Annual Salaries')
  plt.legend()
  plt.savefig(f'{MetaP.REPORT_DIR}/eda_salary_dev_annual_salary_distribution.png')

  # Hourly
  hourly_data = salary_dev[salary_dev['y_true_frequency'] == 'HOURLY']
  min_data = hourly_data['y_true_salary_min'].dropna()
  max_data = hourly_data['y_true_salary_max'].dropna()
  min_data_count_excluded = sum(max_data >= 200)
  max_data_count_excluded = sum(min_data >= 200)
  min_data = min_data[(min_data > 0) & (min_data < 200)]
  max_data = max_data[(max_data > 0) & (max_data < 200)]

  plt.clf()
  plt.hist(min_data, bins=20, alpha=0.5, label='Minimum Salary')
  plt.hist(max_data, bins=20, alpha=0.5, label='Maximum Salary')
  plt.xlabel('Salary')
  plt.ylabel('Frequency')
  plt.title('Salary Distribution - Hourly Wages')
  plt.legend()
  plt.savefig(f'{MetaP.REPORT_DIR}/eda_salary_dev_hourly_salary_distribution.png')


  # Average annual salary by currency
  statistics = ['size', 'min', 'max', 'mean']

  avg_annual_by_currency = salary_dev.dropna(subset=['y_true_currency', 'y_true_salary_min'])
  avg_annual_by_currency = avg_annual_by_currency[avg_annual_by_currency['y_true_frequency'] == 'ANNUAL']
  avg_annual_by_currency = avg_annual_by_currency[avg_annual_by_currency['y_true_salary_min'] > 0]
  avg_annual_by_currency = avg_annual_by_currency.groupby('y_true_currency')['y_true_salary_min'].agg(statistics).reset_index()
  avg_annual_by_currency = avg_annual_by_currency.sort_values(by='mean', ascending=False)
  avg_annual_by_currency[statistics] = avg_annual_by_currency[statistics].astype(int)
  avg_annual_by_currency.to_csv(f'{MetaP.REPORT_DIR}/eda_salary_dev_avg_annual_by_currency.csv', index=False)
  
  avg_hourly_by_currency = salary_dev.dropna(subset=['y_true_currency', 'y_true_salary_min'])
  avg_hourly_by_currency = avg_hourly_by_currency[avg_hourly_by_currency['y_true_frequency'] == 'HOURLY']
  avg_hourly_by_currency = avg_hourly_by_currency[avg_hourly_by_currency['y_true_salary_min'] > 0]
  avg_hourly_by_currency = avg_hourly_by_currency.groupby('y_true_currency')['y_true_salary_min'].agg(statistics).reset_index()
  avg_hourly_by_currency = avg_hourly_by_currency.sort_values(by='mean', ascending=False)
  avg_hourly_by_currency[statistics] = avg_hourly_by_currency[statistics].astype(int)
  avg_hourly_by_currency.to_csv(f'{MetaP.REPORT_DIR}/eda_salary_dev_avg_hourly_by_currency.csv', index=False)

def _summarise_work_arrangement_data(work_arr_dev: pd.DataFrame) -> None:
  """
  Summarise the work arrangement data.

  Args:
    work_arr_dev (pd.DataFrame): The work arrangement data DataFrame.

  Returns:
    None
  """
  # Plot a histogram of work arrangement
  plt.clf()
  work_arr_dev['y_true'].value_counts().plot(kind='bar')
  plt.xlabel('Work Arrangement')
  plt.ylabel('Frequency')
  plt.title('Work Arrangement Distribution')
  plt.savefig(f'{MetaP.REPORT_DIR}/eda_work_arr_dev_distribution.png')


def _summarise_seniority_data(seniority_dev: pd.DataFrame) -> None:
  """
  Summarise the seniority data.

  Args:
    seniority_dev (pd.DataFrame): The seniority data DataFrame.

  Returns:
    None
  """
  # Plot a histogram of seniority
  plt.clf()
  seniority_dev['y_true'].value_counts().plot(kind='bar')
  plt.xlabel('Seniority Level')
  plt.ylabel('Frequency')
  plt.title('Seniority Distribution')
  plt.savefig(f'{MetaP.REPORT_DIR}/eda_seniority_dev_distribution.png')

  seniority_dev['y_true'].value_counts().to_csv(f'{MetaP.REPORT_DIR}/eda_seniority_dev_distribution.csv', index=True)




def _conduct_EDA_salary_data(salary_dev: pd.DataFrame) -> None:
  _summarise_salary_data(salary_dev)

def _conduct_EDA_work_arrangement_data(work_arr_dev: pd.DataFrame) -> None:
  _summarise_work_arrangement_data(work_arr_dev)

def _conduct_EDA_seniority_data(seniority_dev: pd.DataFrame) -> None:
  _summarise_seniority_data(seniority_dev)

def conduct_EDA(data: dict[pd.DataFrame]) -> None:
  _conduct_EDA_salary_data(data['salary_dev'])
  _conduct_EDA_work_arrangement_data(data['work_arr_dev'])
  _conduct_EDA_seniority_data(data['seniority_dev'])

if __name__ == '__main__':
  pass