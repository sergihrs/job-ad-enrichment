import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, dataset, device="cpu", epochs=3, batch_size=16, lr=2e-5):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {ep}/{epochs}", unit="batch"):
            t_ids = batch["text_input_ids"].to(device)
            t_mask = batch["text_attention_mask"].to(device)
            l_ids = batch["label_input_ids"].to(device)
            l_mask = batch["label_attention_mask"].to(device)

            optimizer.zero_grad()
            sims = model(t_ids, t_mask, l_ids, l_mask)
            target = torch.ones_like(sims, device=device)

            # CosineEmbeddingLoss expects inputs (x1,x2, target),
            # so we unsqueeze duplicate sims as a hack:
            loss = criterion(sims.unsqueeze(1), sims.unsqueeze(1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {ep}/{epochs} — avg loss {total_loss/len(loader):.4f}")
