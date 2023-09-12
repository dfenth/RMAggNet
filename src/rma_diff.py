import torch
import numpy as np

class Classifier(torch.nn.Module):
    """A differentiable classifier which approximates the error correction of RMAggNet"""
    def __init__(self, codeword_length, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(codeword_length, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.logits = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Pass data to the model

        Parameters:
        - x (torch.tensor): The input to the model

        Returns:
        - (torch.tensor): The model output as logits
        """
        x = self.fc1(x)
        x = self.fc2(x)

        return self.logits(x)


    def train_data(self, train_data, *, epochs, val_data=None, lr=1e-3, cuda=True):
        """
        Train the data 

        Parameters:
        - train_data (DataLoader): The training data as a torch data loader
        - epochs (int): The number of epochs to train for
        - val_data (DataLoader, optional): The validation data to validate on (default is None)
        - lr (float, optional): The learning rate (default is 1e-3)
        - cuda (bool, optional): Whether CUDA is used for training (default is True)
        """
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.to(device)

        opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            report = "Epoch: {:02}".format(epoch)
            self.train()

            losses = []

            for idx, batch_data in enumerate(train_data):
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)

                opt.zero_grad()

                predictions = self(inputs)
                predictions = torch.squeeze(predictions)
                loss = loss_fn(predictions, labels)

                loss.backward()
                opt.step()

                losses.append(loss.cpu().detach().numpy())
            
            report += " - Loss: {:.4f}".format(np.mean(losses))

            if val_data:
                self.eval()
                correct, total = 0, 0

                for data in val_data:
                    inputs, labels = data
                    inputs = inputs.to(device)

                    predictions = self(inputs)
                    predictions = predictions.cpu().detach().numpy()

                    res = list(map(lambda x: 1 if np.argmax(x[0]) == x[1] else 0, zip(predictions, labels)))

                    correct += sum(res)
                    total += labels.shape[0]

                report += " - Validation Accuracy: {:.4f}".format(correct/total)

            print(report)


def synthetic_dataset_gen(class_codewords, dataset_size):
    """Create a synthetic dataset to train a differentiable thresholding and correction layer
    
    Parameters:
    - class_codewords (list of string): A list of all of the codewords of the classes from the dataset
    - dataset_size (int): The number of synthetic samples to generate

    Returns:
    -dataset (list of (torch.tensor, torch.tensor)): A list of tuples consisting of a tensor (a real valued vector equal to the length of a class codeword) and an integer tensor (the class)
    """
    # We need to generate a random vector of 16 (but actually n) real numbers
    # Threshold them
    # Then compare these to the class codes
    # We set the label as the closest
    # The training example is the random vector and the label!
    codeword_len = len(class_codewords[0])
    
    # Determine the number of codewords per class (to keep input evenly distributed)
    num_samples_per_class = dataset_size//len(class_codewords)
    
    print("INFO: Generating synthetic dataset of {} samples per class for a total dataset size {}"
            .format(num_samples_per_class, num_samples_per_class*len(class_codewords)), "blue")

    # Generate synthetic codewords for each class
    dataset = []
    rng = np.random.default_rng()

    for class_label, class_codeword in enumerate(class_codewords):
        # Generate all samples at once, zeroing specific columns in accordance with the class codeword
        # random returns values within U[0, 1) to change to U[a, b) = (b-a) * U[0,1) + a
        # We want values between [0.5, 1)
        samples = (1-0.5) * rng.random(size=(num_samples_per_class, codeword_len)) + 0.5
        samples = samples.astype(dtype=np.float32) 
        # Zero the bits that need to be zeroed!
        for col_id, bit in enumerate(class_codeword):
            if bit == '0':
                samples[:,col_id] = np.zeros(shape=(num_samples_per_class, ))
        
        samples = torch.tensor(samples)

        # Split the matrix into arrays along the rows
        samples = torch.split(samples, 1)
        labels = [class_label]*num_samples_per_class

        assert len(samples) == len(labels), "Number of samples {} is not equal to the number of labels {}".format(len(samples), len(labels))

        dataset += list(zip(samples, labels))

    return dataset


class RMAggDiff(torch.nn.Module):
    """The differentiable RMAggNet classifier"""
    def __init__(self, rm_aggnet_model):
        super(RMAggDiff, self).__init__()
        self.rm_aggnet = rm_aggnet_model
        self.diff_replacement = Classifier(codeword_length=len(self.rm_aggnet.bitstring_classes[0]), num_classes=len(self.rm_aggnet.class_set))
        
        class_codewords = self.rm_aggnet.bitstring_classes
        train_dataset = synthetic_dataset_gen(class_codewords, 10000)
        val_dataset = synthetic_dataset_gen(class_codewords, 1000)
        batch_size = 128
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.diff_replacement.train_data(train_loader, epochs=10, val_data=val_loader)

    def forward(self, x):
        """
        Pass data to the RMAggNet model, followed by the differentiable classifier

        Parameters:
        - x (torch.tensor): The input to the model

        Returns:
        - (torch.tensor): The model output as logits
        """
        membership_tensor = self.rm_aggnet.forward(x)
        res = self.diff_replacement.forward(membership_tensor)
        return res

    def evaluate(self, data, cuda=True):
        """
        Evaluate the performance of the differentiable model on a dataset

        Parameters:
        - data (Loader): The data to evaluate on
        - cuda (bool, optional): Evaluate using CUDA (default is True)

        Returns:
        - (dict): A dictionary of results
            - "loss" (float): The loss of the model
            - "accuracy" (float): The accuracy on the dataset
        """
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.eval()

        loss_fn = torch.nn.CrossEntropyLoss()
        correct, total = 0, 0
        losses = []

        for idx, batch_data in enumerate(data):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            predictions = self(inputs)
            predictions = torch.squeeze(predictions)
            loss = loss_fn(predictions, labels)
            predictions = predictions.cpu().detach().numpy()

            losses.append(loss.cpu().detach().numpy())

            res = list(map(lambda x: 1 if np.argmax(x[0]) == x[1] else 0, zip(predictions, labels)))

            correct += sum(res)
            total += labels.shape[0]

        return {"loss": np.mean(losses), "accuracy": correct/total}

# Example of application
if __name__ == "__main__":
    batch_size = 64
    class_codewords = ["1111111111111111", 
                       "0000000011111111", 
                       "0101010101010101", 
                       "0011001100110011"]

    train_dataset = synthetic_dataset_gen(class_codewords, 10000)
    val_dataset = synthetic_dataset_gen(class_codewords, 1000)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    classifier = Classifier(codeword_length=16, num_classes=4)
    classifier.train_data(train_loader, epochs=10, val_data=val_loader)
