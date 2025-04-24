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
  VERBOSE: bool = True
  MULTICLASS_OUTPUT_DIR: str = './models/distilbert-finetuned'
  MULTICLASS_OUTPUT_CHECKPOINT_DIR: str = './models/distilbert-finetuned/checkpoint-168'


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
  SENIORITY_KEYWORDS: list[str] = [] # TBD dynamically
  SENIORITY_DEFAULT: str = 'experienced'
  NONE_STR: str = 'None'