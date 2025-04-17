import pandas as pd
from src.models.stat_models.stat_model_functions import stat_model_classifier

def stat_model_seniority(seniority_dev: pd.DataFrame) -> None:
  stat_model_classifier(seniority_dev, 'seniority')

if __name__ == '__main__':
  pass