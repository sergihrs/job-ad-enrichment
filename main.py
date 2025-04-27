from src.data.data import load_data, preprocess_data
from src.EDA import conduct_EDA
from src.models.run_models import run_stat_models, run_fine_tuned_models, run_proprietary_models, run_facebook_opt359m
from src.ancillary_functions import setup, save_user_testing_data
from src.config import MetaP, CMDArgs
from src.models.stat_models.stat_model_seniority import seniority_rule_based
from src.models.stat_models.stat_model_work_arr import stat_model_work_arr_on_test_data
from src.models.stat_models.stat_model_salary import salary_rule_based


def _run_all(data):
  print('Running all models...')
  conduct_EDA(data)
  run_stat_models(data)
  # run_fine_tuned_models(data)
  # run_proprietary_models(data)
  
def _run_one(data):
  print('TESTDATA:')
  print(data['test_data'])
  if CMDArgs.STAT:
    if CMDArgs.TARGET == 'seniority':
      print('Running seniority statistical model...')
      outcome = seniority_rule_based(data['test_data'], do_group=True)
    elif CMDArgs.TARGET == 'work_arr':
      print('Running work_arr statistical model...')
      outcome = stat_model_work_arr_on_test_data(data['work_arr_dev'], data['test_data'])
    else:
      print('Running salary statistical model...')
      outcome = salary_rule_based(data['test_data'])
  else:
    if CMDArgs.TARGET == 'seniority':
      print('Running seniority fine-tuned model...')
      outcome = run_facebook_opt359m(data=data, test_data=data['test_data'], model_name='seniority')
    elif CMDArgs.TARGET == 'work_arr':
      print('Running work_arr fine-tuned model...')
      outcome = run_facebook_opt359m(data=data, test_data=data['test_data'], model_name='work_arr')
    else:
      print(f'Fine tuned mode for {CMDArgs.TARGET} not supported yet')
      outcome = None
      
  if outcome is not None:
    save_user_testing_data(outcome)

def main():
  setup()
  the_data = load_data()
  the_data = preprocess_data(the_data)
  
  if CMDArgs.FILE is None:
    _run_all(the_data)
  else:
    _run_one(the_data)
  
if __name__ == '__main__':
  main()