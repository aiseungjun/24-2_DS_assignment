import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Resize, CenterCrop


def custom_collate_fn(batch):
    videos, labels = zip(*[(item[0], item[2]) for item in batch])
    videos = torch.stack(videos, dim=0)
    labels = torch.tensor(labels)
    return videos, labels


def get_dataloaders(data_dir, annotation_path, batch_size=8, num_workers=4, train_split=0.8):
    """
    Load the UCF101 dataset and create train/validation DataLoaders.
    """
    transform = Compose([
        Resize((112,112)),
        CenterCrop(112),
    ])

    dataset = UCF101(
        root=data_dir,
        annotation_path=annotation_path,
        frames_per_clip=16,
        step_between_clips=1,
        transform=transform
    )

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    return train_loader, val_loader
