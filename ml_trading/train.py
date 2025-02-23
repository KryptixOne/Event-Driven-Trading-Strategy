import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                logits_val = model(X_val)
                loss_val = criterion(logits_val, y_val)
                val_loss += loss_val.item() * X_val.size(0)

                preds = torch.argmax(logits_val, dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

    return model
