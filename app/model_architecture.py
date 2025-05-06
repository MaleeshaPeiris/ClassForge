import torch.nn as nn

class StudentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(StudentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)