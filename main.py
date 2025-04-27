from src.data.data import load_data, preprocess_data
from src.EDA import conduct_EDA
from src.models.run_models import run_stat_models, run_fine_tuned_models
from src.ancillary_functions import setup
from src.config import MetaP, CMDArgs

from src.ancillary_functions import connect_to_anthropic

def main():
  setup()
  
  if MetaP.DO_PARSE_ARGS:
    # Do stuff with the command line arguments
    print(CMDArgs.TARGET)
    print(CMDArgs.FILE)
    print(CMDArgs.STAT)
  else:
    # Run through everything
    the_data = load_data()
    the_data = preprocess_data(the_data)
    conduct_EDA(the_data)
    run_stat_models(the_data)
    run_fine_tuned_models(the_data)
  
  
if __name__ == '__main__':
  main()