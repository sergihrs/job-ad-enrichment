from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
from nltk.corpus import words as nltk_words
import os
from sklearn.model_selection import train_test_split
from src.config import HyperP, MetaP

english_words = set(nltk_words.words())

def clean_html(html_content: pd.Series) -> str:
  """
  Extract text from HTML content.
  
  Args:
      html_content (str): The HTML content to extract text from.
  
  Returns:
      str: The extracted text.
  """
  # soup = BeautifulSoup(html_content, 'html.parser')
  # words = soup.get_text().split()
  # words = [word.strip().lower() for word in words if word.strip()]

  # # Retain only words that are in the English dictionary
  # words = ' '.join([word for word in words if word.lower() in english_words])

  # Remove html tags from job_ad_details  
  html_content_cleaned = html_content.str.replace(
      r"<[^>]+>", " ", regex=True
  )

  # Remove html & characters from job_ad_details
  html_content_cleaned = html_content_cleaned.str.replace(
      r"&[a-zA-Z0-9]+;", " ", regex=True
  )

  # Remove escaped characters from job_ad_details
  html_content_cleaned = html_content_cleaned.str.replace(
      r"\\[a-zA-Z0-9]+", " ", regex=True
  )

  # Merge multiple spaces into one
  html_content_cleaned = html_content_cleaned.str.replace(
      r"\s+", " ", regex=True
  )

  html_content_cleaned = html_content_cleaned.apply(lambda x: x.lower())

  return html_content_cleaned


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
  






def stat_model_classifier(data: pd.DataFrame, dataset_type: str) -> None:
  """
  Fit a statistical model to the d  ata (e.g. seniority or work arrangements).
  
  Args:
      data (pd.DataFrame): The data to fit the model to. Must have 'job_ad_details' and 'y_true' columns.
  
  Returns:
      None: Just saves the accuracy results by y_true classification.
  """ 
  # Model
  train, val = train_test_split(data, test_size=HyperP.VAL_SIZE, random_state=MetaP.RANDOM_SEED)
  model = _train_classifier_on_tf_idf(train['job_ad_details'], train['y_true'])

  # Predict on validation set and report accuracy
  val_predictions = model.predict(val['job_ad_details'])
  val['predictions'] = val_predictions
  val['correct'] = val['y_true'] == val['predictions']
  accuracy_by_level = val.groupby('y_true').agg(
    count=('correct', 'size'),
    sum=('correct', 'sum'),
  )
  accuracy_by_level['accuracy'] = accuracy_by_level['sum'] / accuracy_by_level['count']
  overall_accuracy = val['correct'].sum() / len(val)

  # Save to CSV
  accuracy_by_level.to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'{dataset_type}_stat_based_indiv.csv'),
    index=True
  )
  pd.DataFrame({'overall_accuracy': [overall_accuracy]}).to_csv(
    os.path.join(MetaP.STAT_MODELS_DIR, f'{dataset_type}_stat_based_overall.csv'),
    index=False
  )


if __name__ == '__main__':
  pass