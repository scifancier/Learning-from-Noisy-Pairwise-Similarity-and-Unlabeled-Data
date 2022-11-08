import torch.nn as nn
import torch.nn.functional as F


class NsuNet(nn.Module):
    def __init__(self, input_size=1, num_classes=20):
        super(NsuNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, 1)



    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out

