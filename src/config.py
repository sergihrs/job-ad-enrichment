import argparse

class Parameters:
  @classmethod
  def print(cls):
    for attr, value in cls.__dict__.items():
      if not attr.startswith('__'):
        print(f'{attr} = {value}')
              
class MetaP(Parameters):
  RANDOM_SEED: int = 42
  DATA_DIR: str = 'data'
  REPORT_DIR: str = 'report'
  STAT_MODELS_DIR: str = 'models/stat_models'
  MODELS_DIR: str = 'models'
  VERBOSE: bool = True
  MULTICLASS_OUTPUT_DIR: str = './models/distilbert-finetuned'
  MULTICLASS_OUTPUT_CHECKPOINT_DIR: str = './models/distilbert-finetuned/checkpoint-168'
  Y_TRUE_GROUPING_FILENAME: str = './src/data/y_true_grouping.xlsx'
  DO_PARSE_ARGS: bool = False
  ANTHROPIC_API_KEY: str = None


class HyperP(Parameters):
  VAL_SIZE: float = 0.2
  CURRENCY_CHARS: list[str] = ['$', '€', '£', '¥', '฿', '₱', '₹', '₩', '₺', '₫', '₭', '₮', '₨']
  SALARY_DELIMITER: str = '-'
  FREQ_KEYWORDS: str = {
    'HOURLY': ['per hour', 'hourly', 'p.h', 'ph'],
    'DAILY': ['per day', 'daily', 'p.d', 'pd'],
    'WEEKLY': ['per week', 'weekly', 'p.w'],
    'MONTHLY': ['per month', 'monthly', 'p.m'],
    'ANNUAL': ['per year', 'yearly', 'annually', 'p.a', 'p.y'],
  }
  FREQ_DEFAULT: str = 'MONTHLY'
  COUNTRY_TO_CURR = {
    'AUS': 'AUD',
    'HK': 'HKD',
    'MY': 'MYR',
    'SG': 'SGD',
    'PH': 'PHP',
    'NZ': 'NZD',
    'TH': 'THB',
    'ID': 'IDR',
    'US': 'USD',
  }
  CURR_DEFAULT: str = 'AUD'
  SAL_MIN_MAX_PC_DIFFERENCE: float = 0.6
  SAL_AMOUNT_DEFAULT: int = 0
  SENIORITY_MIN_COUNT: int = 8
  SENIORITY_REPLACE: dict[str, str] = {
    "entry-level": "entry level",
    "mid-senior": "intermediate",
    "mid-level": "intermediate",
    "board": "director",
  }
  SENIORITY_KEYWORDS: list[str] = {
    'junior': ['junior', 'entry level', 'entry-level', 'intern', 'trainee', 'apprentice'],
    'mid-level': ['intermediate', 'mid-level', 'mid-senior', 'mid senior', 'mid level'],
    'senior': ['senior', 'lead', 'principal', 'head', 'chief', 'director', 'manager', 'supervisor', 'team lead', 'team leader', 'executive', 'vp', 'vice president', 'c-level', 'cxo', 'cto', 'ceo', 'coo', 'cfo', 'cmo'],
  }
  SENIORITY_DEFAULT: str = 'experienced'
  NONE_STR: str = 'None'
  FIELD_NAMES_TO_CONSOLIDATE: list[str] = [
    'job_title',
    'job_summary',
    'job_ad_details',
    'classification_name',
    'subclassification_name'
  ]
  DATASETS_FOR_FINE_TUNED_MODELS: dict[tuple[str, str]] = {
    'work_arr': ('work_arr_dev', 'work_arr_test'),
    'seniority': ('seniority_dev', 'seniority_test'),
  }
  DATASETS_AND_PROMPTS_FOR_CLAUDE_MODELS: dict[str, tuple[str, str]] = {
    'work_arr': ('work_arr_test', "Classify the following text as either 'remote', 'onsite' or 'hybrid'. Respond with one word only"),
    'seniority': ('seniority_test', "Classify the following text as either 'junior', 'mid-level' or 'senior'. Respond with one word only"),
    'salary': ('salary_test', "Parse the following job ad into the following format: MIN_PAY-MAX_PAY-CURRENCY-FREQUENCY. Respond in exactly this format. If the job ad does not contain a salary, respond with '0-0-None-None'"),
  }
  FACEBOOK_OPT350M_PROMPTS: dict[str] = {
    'work_arr': 'Work Arrangement',
    'seniority': 'Seniority',
  }
  BERT_PROMPTS: dict[str] = {
    'work_arr': '',
    'seniority': '',
  }
  MAX_TRAINING_STEPS: int = 1
  
class CMDArgs(Parameters):
  TARGET: str = ['seniority', 'work_arr']
  FILE = {
    'seniority': './data/seniority_labelled_test_set.csv',
    'work_arr': './data/work_arrangements_test_sets.csv',
  }
  STAT: bool = False
  
  @classmethod
  def parse_args(cls, args: argparse.Namespace) -> None:
    if args.stat:
      print('Running statistical models...')
      cls.STAT = True
    else:
      print('Running non-statistical models...')
      cls.STAT = False
    
    if args.target is None:
      cls.TARGET = 'seniority'
      print('No target specified. Running "seniority"...')
    else:
      cls.TARGET = args.target
      print(f"Running target: {cls.TARGET}")
    
    if args.file is None:
      cls.FILE = cls.FILE[cls.TARGET]
      print(f'No input file specified. Using default test data {cls.FILE}...')
    else:
      cls.FILE = args.file
      print(f"Will apply model to file: {cls.FILE}")