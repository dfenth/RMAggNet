import torch
import torchvision
import time
import numpy as np

class AbsModel(torch.nn.Module):
    def __init__(self):
        super(AbsModel, self).__init__()

            
    def train_on_data(self, train_dataset, val_dataset=None, epochs=10, lr=1e-3, optimiser=None, verbose=False, logger=None):
        self.to('cuda')
        
        if not optimiser:
            opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        else:
            opt = optimiser

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            self.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            for idx, data in enumerate(train_dataset):
                
                batch_start_time = time.time()
                
                inputs, labels = data
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                opt.zero_grad()
                out = self.forward(inputs)
                #out = torch.nn.functional.softmax(out, dim=1) # Softmax done implicitly in the Cross Entropy loss
                
                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                train_losses.append(loss.to('cpu').detach().numpy())
                
                predictions = out.argmax(dim=1)
                train_total += labels.shape[0]
                train_correct += int((predictions == labels).sum())


                if verbose:
                    if logger:
                        logger.info("Batch {}/{}: {:.3f}s".format(idx, len(train_dataset), time.time()-batch_start_time))
                    else:
                        print("Batch {}/{}: {:.3f}s".format(idx, len(train_dataset), time.time()-batch_start_time), end='\r')
            
            if logger:
                logger.info("Epoch {}: Took {:.3f}s".format(epoch, time.time()-epoch_start_time))
            else:
                print("\nEpoch {}: Took {:.3f}s".format(epoch, time.time()-epoch_start_time))
        
            if val_dataset != None:
                self.eval()
                val_start_time = time.time()
                val_losses = []
                val_correct = 0
                val_total = 0

                for data in val_dataset:
                    inputs, labels = data
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                    out = self.forward(inputs)
                    #out = torch.nn.functional.softmax(out, dim=1)
                    
                    loss = loss_fn(out, labels)
                    val_losses.append(loss.cpu().detach().numpy())

                    predictions = out.argmax(dim=1)
                    val_total += labels.shape[0]
                    val_correct += int((predictions == labels).sum())


                
                if logger:
                    logger.info("Train acc: {:.4f}, Train loss: {:.4f} - Val acc: {:.4f}, Val loss: {:.4f}, Took: {:.3f}s"
                        .format(train_correct/train_total, np.mean(train_losses), val_correct/val_total, np.mean(val_losses), time.time()-val_start_time))
                else:
                    print("Train acc: {:.4f}, Train loss: {:.4f} - Val acc: {:.4f}, Val loss: {:.4f}, Took: {:.3f}s"
                        .format(train_correct/train_total, np.mean(train_losses), val_correct/val_total, np.mean(val_losses), time.time()-val_start_time))
            
                # Update scheduler based on validation results
                scheduler.step(np.mean(val_losses))

    def evaluate(self, test_dataset, logger=None):
        self.eval()
        eval_start_time = time.time()
        correct = 0
        total = 0

        for data in test_dataset:
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            out = self.forward(inputs)
            predictions = out.argmax(dim=1)
            total += labels.shape[0]
            correct += int((predictions == labels).sum())
        
        if logger:
            logger.info("Test set acc: {:.4f} - {:.3f}s"
                    .format(correct/total, time.time()-eval_start_time))
        else:
            print("Test set acc: {:.4f} - {:.3f}s"
                    .format(correct/total, time.time()-eval_start_time))


class MNISTModel(AbsModel):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(7*7*64, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

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

        out = logits
        return out


class EMNISTModel(AbsModel):
    def __init__(self):
        super(EMNISTModel, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 47)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class CIFARModel(AbsModel):
    def __init__(self):
        super(CIFARModel, self).__init__()
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
        self.fc3 = torch.nn.Linear(256, 10)
        

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

    def train_on_data(self, train_dataset, val_dataset=None, epochs=10, lr=1e-3, verbose=False, logger=None):
        # Override train_on_data to provide custom optmiser with heavy weight decay!
        opt = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=0.0025)
        super().train_on_data(train_dataset, val_dataset=val_dataset, epochs=epochs, lr=lr, verbose=verbose, optimiser=opt, logger=logger)


"""
class CIFARModel(AbsModel):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 10)
        
        #self.bn1 = torch.nn.BatchNorm1d(num_features=256, affine=True)

        #self.fc1 = torch.nn.Linear(256, 512)
        #self.fc2 = torch.nn.Linear(512, 256)
        #self.fc3 = torch.nn.Linear(256, 10)
        
        #torch.nn.init.kaiming_uniform_(self.fc1.weight)
        #torch.nn.init.kaiming_uniform_(self.fc2.weight)
        #torch.nn.init.kaiming_uniform_(self.fc3.weight)

        #for param in self.resnet18.parameters():
        #    torch.nn.init.kaiming_uniform_(param.weight)


    def forward(self, x):
        x = self.resnet18(x)
        #x = torch.nn.functional.relu(x)
        #x = torch.nn.Dropout(p=0.7)(x)
        
        #x = self.bn1(x)

        #x = self.fc1(x)
        #x = torch.nn.functional.relu(x)
        #x = torch.nn.Dropout(p=0.8)(x)

        #x = self.fc2(x)
        #x = torch.nn.functional.relu(x)
        #x = torch.nn.Dropout(p=0.9)(x)
        
        #x = self.fc3(x)
        return x
    
    def train_on_data(self, train_dataset, val_dataset=None, epochs=10, lr=1e-3, verbose=False, logger=None):
        # Override train_on_data to provide custom optmiser with heavy weight decay!
        opt = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=0.01)
        super().train_on_data(train_dataset, val_dataset=val_dataset, epochs=epochs, lr=lr, verbose=verbose, optimiser=opt, logger=logger)
"""
