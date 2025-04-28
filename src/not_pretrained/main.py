from src.not_pretrained.data import CSVDataset
from src.not_pretrained.model import TextLabelSimModel
from src.not_pretrained.train import train
import torch


def main():
    dataset = CSVDataset(
        "data/seniority_train.csv", text_col="job_text", label_col="y_true"
    )
    print(f"Dataset loaded. Size: {len(dataset)}")

    model = TextLabelSimModel(
        encoder_name="bert-base-uncased",
        projection_dim=768,
    )
    print(f"Model loaded. Embedding dimension: {model.text_proj.out_features}")

    train(
        model,
        dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=3,
        batch_size=16,
        lr=2e-5,
    )


if __name__ == "__main__":
    main()
