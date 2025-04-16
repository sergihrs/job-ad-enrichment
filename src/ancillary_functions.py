import os
from src.config import MetaP

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
