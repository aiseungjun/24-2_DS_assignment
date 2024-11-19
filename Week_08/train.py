import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import CustomDataset
import os


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize feature vectors
        features = nn.functional.normalize(features, dim=1)

        # Compute similarity scores
        similarity_matrix = torch.mm(features, features.T)
        labels = labels.unsqueeze(1)

        # Compute mask for Positive Pairs
        positive_mask = labels == labels.T

        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # Positive and Negative Pairs
        positives = similarity_matrix[positive_mask].view(-1)
        negatives = similarity_matrix[~positive_mask & ~mask].view(-1)

        # Compute Contrastive Loss
        positive_loss = -torch.log(torch.exp(positives / self.temperature).sum())
        negative_loss = torch.log(torch.exp(negatives / self.temperature).sum())
        loss = (positive_loss + negative_loss) / features.size(0)
        return loss


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze CLIP model weights
for param in clip_model.parameters():
    param.requires_grad = False


classifier_model = CustomCLIPClassifier(clip_model).to(device)

# Optimizers for different steps
optimizer_contrastive = torch.optim.Adam(classifier_model.feature_layer.parameters(), lr=1e-4)
optimizer_classifier = torch.optim.Adam(classifier_model.classifier.parameters(), lr=1e-4)

# Schedulers for learning rate decay
scheduler_contrastive = torch.optim.lr_scheduler.StepLR(optimizer_contrastive, step_size=6, gamma=0.5)
scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=6, gamma=0.5)

# Loss functions
contrastive_loss_fn = ContrastiveLoss()
ce_loss_fn = nn.CrossEntropyLoss()

# Dataset and Dataloader
train_dataset = load_from_disk("/root/Representational-Learning/dataset/dataset/train")
val_dataset = load_from_disk("/root/Representational-Learning/dataset/dataset/val")
train_dataloader = DataLoader(CustomDataset(train_dataset, preprocess), batch_size=64, shuffle=True)
val_dataloader = DataLoader(CustomDataset(val_dataset, preprocess), batch_size=64, shuffle=False)

# Training Step 1: Contrastive Loss
print("Step 1: Training with Contrastive Loss")
classifier_model.train()
for epoch in range(30):  # 50 epochs for contrastive loss
    total_loss = 0
    for images, labels in tqdm(train_dataloader, desc=f"Contrastive Epoch {epoch + 1}/50"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass for contrastive loss
        features = classifier_model(images, return_features=True)

        # Compute contrastive loss
        contrastive_loss = contrastive_loss_fn(features, labels)

        # Backward pass
        optimizer_contrastive.zero_grad()
        contrastive_loss.backward()
        optimizer_contrastive.step()

        total_loss += contrastive_loss.item()

    # Step the scheduler
    scheduler_contrastive.step()

    print(f"Epoch {epoch + 1}, Contrastive Loss: {total_loss / len(train_dataloader):.4f}")

# Training Step 2: Cross Entropy Loss
print("Step 2: Training with Cross Entropy Loss")
for epoch in range(30):  # 50 epochs for cross entropy loss
    total_loss = 0
    classifier_model.train()

    for images, labels in tqdm(train_dataloader, desc=f"CE Epoch {epoch + 1}/50"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass for CE loss
        logits = classifier_model(images)

        # Compute CE loss
        ce_loss = ce_loss_fn(logits, labels)

        # Backward pass
        optimizer_classifier.zero_grad()
        ce_loss.backward()
        optimizer_classifier.step()

        total_loss += ce_loss.item()

    # Step the scheduler
    scheduler_classifier.step()

    print(f"Epoch {epoch + 1}, CE Loss: {total_loss / len(train_dataloader):.4f}")

# Save the trained model
model_save_path = "/saved_model/model_last.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
