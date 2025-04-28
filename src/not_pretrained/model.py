import torch.nn as nn
from transformers import AutoModel


class TextLabelSimModel(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased", projection_dim=768):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name)
        self.label_encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.text_encoder.config.hidden_size

        self.text_proj = nn.Linear(hidden_size, projection_dim)
        self.label_proj = nn.Linear(hidden_size, projection_dim)

        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(
        self, text_input_ids, text_attention_mask, label_input_ids, label_attention_mask
    ):
        t_out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
        )
        t_vec = t_out.pooler_output  # (batch, hidden)
        t_proj = self.text_proj(t_vec)  # (batch, proj_dim)

        l_out = self.label_encoder(
            input_ids=label_input_ids,
            attention_mask=label_attention_mask,
            return_dict=True,
        )
        l_vec = l_out.pooler_output  # (batch, hidden)
        l_proj = self.label_proj(l_vec)  # (batch, proj_dim)

        sim = self.cosine(t_proj, l_proj)  # (batch,)
        return sim
