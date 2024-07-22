import torch
import torch.nn as nn
import torch.nn.functional as F

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):    
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class LCNNFull(nn.Module):
    def __init__(self):
        super(LCNNFull, self).__init__()

        # First part from previous implementation
        self.conv1 = mfm(1, 32, 5, 1, 1)  # (32, 5, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = mfm(32, 32, 1, 1, 1)  # (32, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = mfm(32, 48, 3, 1, 1)  # (48, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = mfm(48, 48, 1, 1, 1)  # (48, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = mfm(48, 64, 3, 1, 1)  # (64, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = mfm(64, 64, 1, 1, 1)  # (64, 1, 1, 1)
        self.bn6 = nn.BatchNorm2d(64)

        
        self.conv7 = mfm(64, 32, 3, 1, 1)  # (32, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(32)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = mfm(32, 64, 1, 1, 1)  # (64, 1, 1, 1)
        self.bn8 = nn.BatchNorm2d(64)

        self.conv9 = mfm(64, 32, 3, 1, 1)  # (32, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(32)
        self.pool9 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = mfm(32, 32, 1, 1, 1)  # (32, 1, 1, 1)
        self.bn10 = nn.BatchNorm2d(32)

        self.conv11 = mfm(32, 32, 3, 1, 1)  # (32, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(32)
        self.dropout11 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_final = nn.BatchNorm2d(32)
        self.dropout_final = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 2)  # Assuming 2 classes for the output

    def forward(self, x):
        # First part
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.pool7(x)

        x = self.conv8(x)
        x = self.bn8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.pool9(x)

        x = self.conv10(x)
        x = self.bn10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.dropout11(x)

        x = self.global_avg_pool(x)
        x = self.bn_final(x)
        x = self.dropout_final(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def LightCNN():
    model = LCNNFull()
    return model
