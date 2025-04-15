from src.data import load_data, preprocess_data
from src.EDA import conduct_EDA

def main():
  the_data = load_data()
  the_data = preprocess_data(the_data)
  conduct_EDA(the_data)
  
if __name__ == '__main__':
  main()