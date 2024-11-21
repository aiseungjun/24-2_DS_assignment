import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch with progress bar.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(dataloader, desc="Training", unit="batch") as progress_bar:
        for inputs, labels in progress_bar:
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                "Loss": f"{total_loss / (total + 1):.4f}",
                "Accuracy": f"{100.0 * correct / (total + 1):.2f}%",
            })

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy
