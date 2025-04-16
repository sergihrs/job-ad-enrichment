from src.data import load_data, preprocess_data
from src.EDA import conduct_EDA
from src.models.statistical_models import statistical_model_seniority
from src.ancillary_functions import setup

def main():
  setup()
  the_data = load_data()
  the_data = preprocess_data(the_data)
  # conduct_EDA(the_data)
  statistical_model_seniority(seniority_dev=the_data['seniority_dev'])
  
if __name__ == '__main__':
  main()