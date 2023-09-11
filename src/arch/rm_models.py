import torch
import torchvision

class RMMNISTModel(torch.nn.Module):
    def __init__(self):
        super(RMMNISTModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(7*7*64, 1024)
        self.fc2 = torch.nn.Linear(1024, 1)

    def forward(self, x):
        # Using the architecture from the CCAT paper
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        logits = self.fc2(x)
        sigmoid = torch.nn.functional.sigmoid(logits)
        out =  sigmoid
        return out

class RMEMNISTModel(torch.nn.Module):
    def __init__(self):
        super(RMEMNISTModel, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, x):
        #x = self.inp_layer(x)
        logits = self.resnet18(x)
        sigmoid = torch.nn.functional.sigmoid(logits)
        return sigmoid

class RMCIFARModel(torch.nn.Module):
    def __init__(self):
        super(RMCIFARModel, self).__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
        )
        
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
        )
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
        )

        
        self.fc1 = torch.nn.Linear(2048, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        

    def forward(self, x):
        x = self.block1(x)
        x = torch.nn.MaxPool2d(kernel_size=2)(x)
        x = torch.nn.Dropout(p=0.4)(x)
        
        x = self.block2(x)
        x = torch.nn.MaxPool2d(kernel_size=2)(x)
        x = torch.nn.Dropout(p=0.5)(x)
        
        x = self.block3(x)
        x = torch.nn.MaxPool2d(kernel_size=2)(x)
        x = torch.nn.Dropout(p=0.6)(x)
        
        x = torch.nn.Flatten()(x)
        
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.Dropout(p=0.7)(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.Dropout(p=0.8)(x)
        x = self.fc3(x)
        
        return x

