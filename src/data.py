from datasets import load_dataset
import pandas as pd
import os


DATA_PATH = "data/"


def preprocess_seniority():
    """
    Preprocesses the seniority dataset and save train and test datasets.
    """

    df_train = pd.read_csv(f"{DATA_PATH}/seniority_labelled_development_set.csv")
    df_test = pd.read_csv(f"{DATA_PATH}/seniority_labelled_test_set.csv")

    # Remove html tags from job_ad_details
    df_train["job_ad_details"] = df_train["job_ad_details"].str.replace(
        r"<[^>]+>", "", regex=True
    )

    # Remove &nbsp; and \n from job_ad_details
    df_train["job_ad_details"] = df_train["job_ad_details"].str.replace(
        "&nbsp;", "", regex=False
    )
    df_train["job_ad_details"] = df_train["job_ad_details"].str.replace(
        "\n", "", regex=False
    )

    # Merge multiple spaces into one
    df_train["job_ad_details"] = df_train["job_ad_details"].str.replace(
        r"\s+", " ", regex=True
    )

    # Merge columns that mean the same
    df_train["y_true"].replace(
        {
            "entry-level": "entry level",
            "mid-senior": "intermediate",
            "mid-level": "intermediate",
            "board": "director",
        },
        inplace=True,
    )

    # Filter columns with less than 8 occurrences
    df_train["value_count"] = df_train["y_true"].map(df_train["y_true"].value_counts())
    df_train = df_train[df_train["value_count"] >= 8]

    # Merge all text inputs into one column
    merge_text = lambda x: (
        x["job_title"]
        + ". "
        + x["job_summary"]
        + ". "
        + x["job_ad_details"]
        + ". "
        + x["classification_name"]
        + ". "
        + x["subclassification_name"]
    )
    df_train["job_text"] = df_train.apply(merge_text, axis=1)
    df_test["job_text"] = df_test.apply(merge_text, axis=1)

    # Save to csv files. Only text and y_true
    df_train = df_train[["job_text", "y_true"]]
    df_test = df_test[["job_text", "y_true"]]

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    df_train.to_csv(DATA_PATH + "seniority_train.csv", index=False)
    df_test.to_csv(DATA_PATH + "seniority_test.csv", index=False)


def load_data_mc(dataset: str = "seniority"):
    """
    Loads the dataset for multiclass classification.
    This function loads the dataset from the specified path and returns the train and test datasets.

    Args:
        dataset (str): "seniority", "work_arrangements", or "salary".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    # Load the dataset
    data_files = {
        "train": DATA_PATH + f"{dataset}_train.csv",
        "test": DATA_PATH + f"{dataset}_test.csv",
    }
    assert all(
        [os.path.exists(file) for file in data_files.values()]
    ), "Dataset files not found. Please run the preprocessing script."

    dataset = load_dataset("csv", data_files=data_files, delimiter=",")

    return dataset["train"], dataset["test"]


if __name__ == "__main__":
    # Preprocess the seniority dataset
    preprocess_seniority()

    # Load the dataset
    train_dataset, test_dataset = load_data_mc("seniority")
    print("Train dataset:", train_dataset)
    print("Test dataset:", test_dataset)

    #  Iterate over the train dataset and print the first 5 samples
    for i, sample in enumerate(train_dataset):
        print(f"Sample {i}: {sample}")
        if i == 4:
            break
