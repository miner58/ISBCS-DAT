import torch.nn as nn

#namely The Subject Classifier SD
class DomainClassifier(nn.Module):
    def __init__(self, input_dim =64, output_dim=14):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.classifier(x)
        return x