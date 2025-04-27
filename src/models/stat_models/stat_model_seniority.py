import pandas as pd
from src.models.stat_models.stat_model_functions import clean_html, stat_model_classifier
from src.config import HyperP, MetaP
import os
import re


def _get_stat_model_seniority(seniority_text: str) -> None:
  for level, keywords in HyperP.SENIORITY_KEYWORDS.items():
    for keyword in keywords:
      if keyword in seniority_text:
        return level
          
  return HyperP.SENIORITY_DEFAULT


def seniority_rule_based(seniority_dev: pd.DataFrame, do_group: bool=False) -> None:
  """Predict the frequency of salary based on the text.
  Args:
    salary_dev (pd.DataFrame): The full dev dataset to predict the frequency from.

  Returns:
    pd.Series[str]: The predicted frequency.
  """
  predictions = seniority_dev['consolidated_fields'].apply(_get_stat_model_seniority)
  
  if do_group:
    seniority_dev['actual'] = seniority_dev['y_true_grouped']
    seniority_grouping = pd.read_excel('./src/data/y_true_grouping.xlsx', sheet_name='seniority')
    # Apply the mapping to predictions and to seniority_dev['y_true_merged']
    mapping_dict = dict(zip(seniority_grouping['y_true'], seniority_grouping['y_true_grouped']))
    predictions = predictions.map(mapping_dict)   
  else:
    seniority_dev['actual'] = seniority_dev['y_true']
  
  seniority_dev['pred'] = predictions
  seniority_dev['correct'] = seniority_dev['pred'] == seniority_dev['actual']
  accuracy_by_seniority = seniority_dev.groupby('actual').agg(
    count=('actual', 'size'),
    sum=('correct', 'sum'),
  )
  accuracy_by_seniority['accuracy'] = accuracy_by_seniority['sum'] / accuracy_by_seniority['count']
  overall_accuracy = seniority_dev['correct'].sum() / len(seniority_dev)

  # Save to CSV
  accuracy_by_seniority.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'seniority_rule_based_grouped_{do_group}_indiv.csv'),
    index=True
  )
  pd.DataFrame({'overall_accuracy': [overall_accuracy]}).to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'seniority_rule_based_grouped_{do_group}_overall.csv'),
    index=False
  )
  return seniority_dev


def stat_model_seniority(seniority_dev: pd.DataFrame) -> None:
  stat_model_classifier(seniority_dev, 'seniority')
  seniority_rule_based(seniority_dev, do_group=False)
  seniority_rule_based(seniority_dev, do_group=True)
  

if __name__ == '__main__':
  pass