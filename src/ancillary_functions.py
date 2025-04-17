import os
from src.config import MetaP

def custom_round(x):
  import math
  frac = x - int(x)
  if frac <= 0.5:
    return math.floor(x)
  else:
    return math.ceil(x)

def setup():
  # Directories
  folders = [
    MetaP.DATA_DIR,
    MetaP.REPORT_DIR
  ]

  for folder in folders:
    os.makedirs(folder, exist_ok=True)

if __name__ == '__main__':
  pass
