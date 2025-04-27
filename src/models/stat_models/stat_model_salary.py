from src.models.stat_models.stat_model_functions import clean_html
from src.config import MetaP, HyperP
import re
import pandas as pd
import spacy
from src.ancillary_functions import custom_round
import os

nlp = spacy.load('en_core_web_sm')


def _get_salary_text(salary_dev: pd.DataFrame) -> str:
  salary_dev['job_ad_details'] = clean_html(salary_dev['job_ad_details'])
  
  text = (
    (
        salary_dev['job_title']
        + ' '
        + salary_dev['job_ad_details']
        + ' '
        + salary_dev['salary_additional_text']
    )
    .astype(str)
    .str.strip()
    .str.lower()
  )

  return text

def _get_salary_freq_rule_based(salary_text: str) -> None:
  for freq, keywords in HyperP.FREQ_KEYWORDS.items():
    for keyword in keywords:
      # Use regex to match \bkeyword\b to avoid partial matches
      if re.search(rf'\b{keyword}\b', salary_text):
        return freq
          
  return HyperP.FREQ_DEFAULT


def _salary_freq_rule_based(salary_dev: pd.DataFrame) -> None:
  """Predict the frequency of salary based on the text.
  Args:
    salary_dev (pd.DataFrame): The full dev dataset to predict the frequency from.

  Returns:
    pd.Series[str]: The predicted frequency.
  """
  text = _get_salary_text(salary_dev)
  frequencies = text.apply(_get_salary_freq_rule_based)
  return frequencies


def _salary_curr_rule_based(salary_dev: pd.DataFrame) -> str:
  """Predict the currency of salary based on the country code.
  Args:
    x (pd.DataFrame): The salary dev data

  Returns:
    str: The predicted currency.
  """
  country_code = salary_dev['nation_short_desc'].str.strip().str.upper()

  # Check if the country code is in the dictionary
  currency_code = country_code.apply(
    lambda x: HyperP.COUNTRY_TO_CURR[x] if x in HyperP.COUNTRY_TO_CURR else HyperP.CURR_DEFAULT
  )

  return currency_code


def _get_salary_amount_rule_based(text: str) -> float:
  """Predict the amount of salary based on the text.
  Args:
    x (pd.Series): The row of the dataframe containing the text.

  Returns:
    float: The predicted amount.
  """
  # Define a regex pattern to match salary frequency keywords and capture 20 characters before
  for freq, keywords in HyperP.FREQ_KEYWORDS.items():
    for keyword in keywords:
      # Regex to capture 20 characters before the keyword
      pattern = rf'(.{{0,20}})\b{keyword}\b'
      match = re.search(pattern, text)
      if match:
        # Extract the 20 characters before the keyword
        context_before_keyword = match.group(1).strip()
        # Use spacy to extract the numbers before the keyword
        
        doc = nlp(context_before_keyword)
        numbers = [token.text for token in doc if token.like_num and any(char.isdigit() for char in token.text)]

        # Parse numbers to float and filter out non-positive numbers
        numbers = [float(num.replace(',', '')) for num in numbers]
        numbers = [num for num in numbers if num > 0]

        if numbers:
          # Get the index of the largest number
          max_index = numbers.index(max(numbers))
          # Get the corresponding number
          max_salary_amount = numbers[max_index]
          
          if max_index > 0:
            # Assume the number just prior to max_index is the min salary
            salary_amount = numbers[max_index - 1]
            # Consider this salary to be a valid minimum salary if it is with SAL_MIN_MAX_PC_DIFFERENCE
            # of the max salary
            if abs(max_salary_amount - salary_amount) / max_salary_amount < HyperP.SAL_MIN_MAX_PC_DIFFERENCE:
              min_salary_amount = salary_amount
            else:
              min_salary_amount = max_salary_amount

          else:
            min_salary_amount = max_salary_amount

          # Round down if .5 or less, round up if more than .5
          min_salary_amount = custom_round(min_salary_amount)
          max_salary_amount = custom_round(max_salary_amount)

          return min_salary_amount, max_salary_amount

        
      return HyperP.SAL_AMOUNT_DEFAULT, HyperP.SAL_AMOUNT_DEFAULT

  return 0.0  # Default value if no match is found

def _salary_amount_rule_based(salary_dev: pd.DataFrame) -> None:
  """Predict the salary amount based on the text.
  Args:
    salary_dev (pd.DataFrame): The full dev dataset to predict the salary amount from.

  Returns:
    pd.Series[(float, float)]: The predicted min and max salary amounts.
  """
  text = _get_salary_text(salary_dev)
  salaries = text.apply(_get_salary_amount_rule_based)
  return salaries


def salary_rule_based(salary_dev: pd.DataFrame) -> None:
  """Predict the salary based on the text.
  Args:
    salary_dev (pd.DataFrame): The full dev dataset to predict the salary from.

  Returns:
    pd.DataFrame: The predicted salary.
  """
   # Predict frequency
  salary_dev['y_pred_frequency'] = _salary_freq_rule_based(salary_dev)

  # Predict currency
  salary_dev['y_pred_currency'] = _salary_curr_rule_based(salary_dev)

  # Predict amount
  salary_min_and_max = _salary_amount_rule_based(salary_dev)
  salary_dev['y_pred_salary_min'] = salary_min_and_max.apply(lambda x: x[0])
  salary_dev['y_pred_salary_max'] = salary_min_and_max.apply(lambda x: x[1])

  # Calculate MSE on min and max salary predictions
  y_salary_min_mse = sum((salary_dev['y_pred_salary_min'] - salary_dev['y_true_salary_min']) ** 2)
  y_salary_max_mse = sum((salary_dev['y_pred_salary_max'] - salary_dev['y_true_salary_max']) ** 2)

  # Calculate literal accuracy on min and max salaries, currency and frequency
  y_salary_min_accuracy = sum(salary_dev['y_pred_salary_min'] == salary_dev['y_true_salary_min']) / len(salary_dev)
  y_salary_max_accuracy = sum(salary_dev['y_pred_salary_max'] == salary_dev['y_true_salary_max']) / len(salary_dev)
  y_currency_accuracy = sum(salary_dev['y_pred_currency'] == salary_dev['y_true_currency']) / len(salary_dev)
  y_frequency_accuracy = sum(salary_dev['y_pred_frequency'] == salary_dev['y_true_frequency']) / len(salary_dev)

  # Calculate per-category accuracy for currency and frequency
  salary_dev['correct'] = salary_dev['y_pred_currency'] == salary_dev['y_true_currency']
  accuracy_by_currency = salary_dev.groupby('y_true_currency').agg(
    count=('correct', 'size'),
    sum=('correct', 'sum'),
  )
  accuracy_by_currency['accuracy'] = accuracy_by_currency['sum'] / accuracy_by_currency['count']
  salary_dev['correct'] = salary_dev['y_pred_frequency'] == salary_dev['y_true_frequency']
  accuracy_by_frequency = salary_dev.groupby('y_true_frequency').agg(
    count=('correct', 'size'),
    sum=('correct', 'sum'),
  )
  accuracy_by_frequency['accuracy'] = accuracy_by_frequency['sum'] / accuracy_by_frequency['count']


  # Combine predictions into a single column

  # For overall prediction, revert predictions for frequency and currency to 'None' if the amounts are zero
  salary_dev.loc[
    (salary_dev['y_pred_salary_min'] == 0) & (salary_dev['y_pred_salary_max'] == 0),
    ['y_pred_frequency_for_overall', 'y_pred_currency_for_overall']
  ] = HyperP.NONE_STR

  salary_dev['y_pred'] = (
    salary_dev['y_pred_salary_min'].astype(str)
    + HyperP.SALARY_DELIMITER
    + salary_dev['y_pred_salary_max'].astype(str)
    + HyperP.SALARY_DELIMITER
    + salary_dev['y_pred_currency_for_overall'].astype(str)
    + HyperP.SALARY_DELIMITER
    + salary_dev['y_pred_frequency_for_overall'].astype(str)
  )  

  overall_accuracy = sum(salary_dev['y_pred'] == salary_dev['y_true']) / len(salary_dev)

  # Save to CSV
  overall_stats = pd.DataFrame({
    'stat': ['y_salary_min_mse', 'y_salary_max_mse', 'y_salary_min_accuracy', 'y_salary_max_accuracy', 'y_currency_accuracy', 'y_frequency_accuracy', 'overall_accuracy'],
    'value': [y_salary_min_mse, y_salary_max_mse, y_salary_min_accuracy, y_salary_max_accuracy, y_currency_accuracy, y_frequency_accuracy, overall_accuracy]
  })

  overall_stats.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'salary_overall_accuracy.csv'),
    index=True
  )

  accuracy_by_currency.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'salary_rule_based_currency_accruracy.csv'),
    index=True
  )

  accuracy_by_frequency.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'salary_rule_based_freq_accuracy.csv'),
    index=True
  )

  for_output = salary_dev[['y_true_salary_min', 'y_true_salary_max', 'y_true_currency', 'y_true_frequency', 'y_true', \
    'y_pred_salary_min', 'y_pred_salary_max', 'y_pred_currency', 'y_pred_frequency', 'y_pred']]
  
  for_output.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'salary_rule_based_predictions.csv'),
    index=False
  )
    
  return for_output

  
def stat_model_salary(salary_dev: pd.DataFrame) -> None:
  salary_rule_based(salary_dev)
  

if __name__ == '__main__':
  pass