import torch
import torch.nn as nn


__all__ = ['CNN']


class CNN(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2),   # N*50*50
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # N*25*25


            nn.Conv2d(128, 512, kernel_size=3, padding=1),           # N*25*25
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # N*12*12

            nn.Conv2d(512, 512, kernel_size=3, padding=1),          # N*12*12
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.Conv2d(512, 1024, kernel_size=3, padding=1),          # N*12*12
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # N*6*6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# x = torch.randn(1, 1, 50, 50)
# model = CNN(num_classes=31)
# print(model(x).shape)