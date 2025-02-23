import torch
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', save_path="model_checkpoint.pth"):
    """
    Trains the given model using PyTorch, with validation monitoring, tqdm progress bar,
    and checkpoint saving.

    Parameters:
    - model: PyTorch model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - epochs: Number of epochs
    - lr: Learning rate
    - device: 'cpu' or 'cuda'
    - save_path: Path to save the model checkpoints

    Returns:
    - Best-trained model (lowest validation loss)
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    best_val_loss = float('inf')  # Track the best validation loss for checkpointing

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Use tqdm for progress visualization
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

                pbar.set_postfix(train_loss=loss.item())  # Update tqdm display
                pbar.update(1)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation Phase
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
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        }
        torch.save(checkpoint, f"{save_path}_epoch{epoch + 1}.pth")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✔ Best model saved at epoch {epoch + 1} with val loss {avg_val_loss:.4f}")

    print("\nTraining complete. Best model saved at:", save_path)
    return model, train_losses, val_losses, val_accuracies
