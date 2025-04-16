from src.data import load_data, preprocess_data
from src.EDA import conduct_EDA
from src.models.stat_models import stat_model_seniority, stat_model_work_arr
from src.ancillary_functions import setup

def main():
  setup()
  the_data = load_data()
  the_data = preprocess_data(the_data)
  conduct_EDA(the_data)
  stat_model_seniority(seniority_dev=the_data['seniority_dev'])
  stat_model_work_arr(work_arr_dev=the_data['work_arr_dev'])
  
if __name__ == '__main__':
  main()