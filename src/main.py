from src.data import load_data, preprocess_data
from src.EDA import conduct_EDA
from src.models.run_models import run_stat_models
from src.ancillary_functions import setup
import pandas as pd
from src.models.multiclass.train import train as train_multiclass

def main():
  setup()
  the_data = load_data()
  the_data = preprocess_data(the_data)
  # conduct_EDA(the_data)
  run_stat_models(the_data)
  # train_multiclass(the_data)
  
if __name__ == '__main__':
  main()