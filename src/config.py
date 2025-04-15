class Parameters:
  @classmethod
  def print(cls):
    for attr, value in cls.__dict__.items():
      if not attr.startswith("__"):
        print(f"{attr} = {value}")
              
class MetaP(Parameters):
  RANDOM_SEED: int = 42
  DATA_DIR: str = 'data'
  REPORT_DIR: str = 'report'
  VERBOSE: bool = True


class HyperP(Parameters):
  VAL_SIZE: float = 0.2
  CURRENCY_CHARS: list[str] = ['$', '€', '£', '¥', '฿', '₱', '₹', '₩', '₺', '₫', '₭', '₮', '₨']