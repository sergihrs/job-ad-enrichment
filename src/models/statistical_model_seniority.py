import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.config import MetaP, HyperP
from bs4 import BeautifulSoup
import re
from nltk.corpus import words as nltk_words
import os

english_words = set(nltk_words.words())

def _get_words_from_html(html_content: str) -> str:
  """
  Extract text from HTML content.
  
  Args:
      html_content (str): The HTML content to extract text from.
  
  Returns:
      str: The extracted text.
  """
  soup = BeautifulSoup(html_content, 'html.parser')
  words = soup.get_text().split()
  words = [word.strip().lower() for word in words if word.strip()]

  # Retain only words that are in the English dictionary
  words = ' '.join([word for word in words if word.lower() in english_words])
  return words


def _train_classifier_on_tf_idf(x_job_ad_details: pd.Series, y: pd.Series) -> None:
    """
    Train a classifier on TF-IDF features, treating each unique value of y_true as a single document.
    
    Args:
        x_job_ad_details (pd.Series): The feature data (job ad details).
        y (pd.Series): The target data (seniority levels).
    
    Returns:
        None
    """
    # Aggregate job ad details by unique values of y
    aggregated_data = x_job_ad_details.groupby(y).apply(lambda texts: ' '.join(texts))
    aggregated_labels = aggregated_data.index  # Unique y_true values

    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(aggregated_data)
    classifier = LogisticRegression()
    classifier.fit(tfidf_matrix, aggregated_labels)

    model = make_pipeline(vectorizer, classifier)
    return model
  



def statistical_model_seniority(seniority_dev: pd.DataFrame) -> None:
  """
  Fit a statistical model to the seniority data.
  
  Args:
      seniority_dev (pd.DataFrame): The seniority data to fit the model to.
  
  Returns:
      None: Just prints the accuracy results by seniority level.
  """
  # Preprocess
  seniority_dev['job_ad_details'] = seniority_dev['job_ad_details'].apply(lambda x: _get_words_from_html(x))
  
  # Model
  train, val = train_test_split(seniority_dev, test_size=HyperP.VAL_SIZE, random_state=MetaP.RANDOM_SEED)
  model = _train_classifier_on_tf_idf(train['job_ad_details'], train['y_true'])

  # Predict on validation set and report accuracy
  val_predictions = model.predict(val['job_ad_details'])
  val['predictions'] = val_predictions
  val['correct'] = val['y_true'] == val['predictions']
  accuracy_by_seniority = val.groupby('y_true')['correct'].mean()
  overall_accuracy = val['correct'].sum() / len(val)

  # Save to CSV
  accuracy_by_seniority.to_csv(
    os.path.join(MetaP.REPORT_DIR, 'seniority_stat_model_accuracy_by_seniority.csv'),
    index=True
  )
  pd.DataFrame({'overall_accuracy': [overall_accuracy]}).to_csv(
    os.path.join(MetaP.REPORT_DIR, 'seniority_stat_model_overall_accuracy.csv'),
    index=False
  )



  
  

if __name__ == '__main__':
  pass