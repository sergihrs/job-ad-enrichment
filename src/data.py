from datasets import load_dataset


def load_data_mc(path: str = "data/") -> tuple[dict, dict]:
    """
    Loads the dataset for multiclass classification.
    This function loads the dataset from the specified path and returns the train and test datasets.

    Args:
        path (str): The path to the dataset. Default is "data/".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    # Load the dataset
    data_files = {"data": path}
    raw_dataset = load_dataset("csv", data_files=data_files)

    # Split the dataset into train and test sets
    dataset = raw_dataset["data"].train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    return train_dataset, test_dataset
