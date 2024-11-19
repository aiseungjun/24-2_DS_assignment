import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=90):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.feature_layer = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)  # Assuming 512 is the embedding size, 90 classes. adjust accordingly

    def forward(self, images, return_features=False):
        # Extract features using CLIP's image encoder
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        features = features.float()
        
        # Train feature_layer
        if return_features:  # train with CL
            features = self.feature_layer(features)
            return features
        # Train classifier
        else:
            with torch.no_grad():  # train with CE
                features = self.feature_layer(features)
            return self.classifier(features)