import pandas as pd
import re

def _print_all_potential_currency_chars(salary_dev: pd.DataFrame):
  """
  Get all unique currency characters from the salary data.

  Args:
    salary_data (pd.DataFrame): The salary data DataFrame.

  Returns:
    set: A set of unique currency characters.
  """
  all_potential_currency_chars = set()
  for currency in salary_dev['salary_additional_text'].dropna():
    # Use regex to find all currency characters in the string
    currency = set(re.findall(r'[^\w\s]', currency))
    # Add the found currency characters to the set
    all_potential_currency_chars.update(currency)

  print(f'All potential currency characters: {all_potential_currency_chars}')


def conduct_EDA(data: dict[pd.DataFrame]) -> None:
  _print_all_potential_currency_chars(data['salary_dev'])

if __name__ == '__main__':
  pass