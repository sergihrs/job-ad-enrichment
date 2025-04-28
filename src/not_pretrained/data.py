# 1. Dataset returns token IDs for both text and label
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CSVDataset(Dataset):
    def __init__(
        self,
        csv_file,
        text_col="text",
        label_col="label",
        encoder_name="bert-base-uncased",
        max_length=256,
    ):
        df = pd.read_csv(csv_file)
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # tokenize text
        t_enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # tokenize label
        l_enc = self.tokenizer(
            self.labels[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "text_input_ids": t_enc.input_ids.squeeze(0),
            "text_attention_mask": t_enc.attention_mask.squeeze(0),
            "label_input_ids": l_enc.input_ids.squeeze(0),
            "label_attention_mask": l_enc.attention_mask.squeeze(0),
        }
