import pandas as pd
from src.models.stat_models.stat_model_functions import clean_html, stat_model_classifier
from src.config import HyperP, MetaP
import os
import re



def _get_seniority_text(salary_dev: pd.DataFrame) -> str:
  salary_dev['job_ad_details'] = clean_html(salary_dev['job_ad_details'])
  
  text = (
    (
        salary_dev['job_title']
        + ' '
        + salary_dev['job_summary']
        + ' '
        + salary_dev['job_ad_details']
    )
    .astype(str)
    .str.strip()
    .str.lower()
  )

  return text


def _get_stat_model_seniority(seniority_text: str) -> None:
  for keyword in HyperP.SENIORITY_KEYWORDS:
    if keyword in seniority_text:
      return keyword
          
  return HyperP.SENIORITY_DEFAULT


def _stat_model_seniority_lookup(seniority_dev: pd.DataFrame) -> None:
  """Predict the frequency of salary based on the text.
  Args:
    salary_dev (pd.DataFrame): The full dev dataset to predict the frequency from.

  Returns:
    pd.Series[str]: The predicted frequency.
  """
  text = _get_seniority_text(seniority_dev)
  predictions = text.apply(_get_stat_model_seniority)
  seniority_dev['correct'] = predictions == seniority_dev['y_true_merged']
  accuracy_by_seniority = seniority_dev.groupby('y_true_merged').agg(
    count=('y_true_merged', 'size'),
    sum=('correct', 'sum'),
  )
  accuracy_by_seniority['accuracy'] = accuracy_by_seniority['sum'] / accuracy_by_seniority['count']
  overall_accuracy = seniority_dev['correct'].sum() / len(seniority_dev)

  # Save to CSV
  accuracy_by_seniority.to_csv(
    os.path.join(MetaP.REPORT_DIR, f'STATMODEL1 seniority_lookup_individual_accuracy.csv'),
    index=True
  )
  pd.DataFrame({'overall_accuracy': [overall_accuracy]}).to_csv(
    os.path.join(MetaP.REPORT_DIR, f'STATMODEL2 seniority_lookup_overall_accuracy.csv'),
    index=False
  )
  return seniority_dev


def stat_model_seniority(seniority_dev: pd.DataFrame) -> None:
  stat_model_classifier(seniority_dev, 'seniority')
  _stat_model_seniority_lookup(seniority_dev)
  

if __name__ == '__main__':
  pass