import torch
import numpy as np

import time
import os
import pickle
import copy

import utils.reed_muller_codes as rmc
import utils.hamming_utils as ham
import utils.progress_bar as progress_bar

class MembershipModel(torch.nn.Module):
    """A classification model for determining set membership"""
    
    def __init__(self, class_list, model):
        """
        Initialise the module with a model that will be used to determine set membership
        
        Parameters:
        - class_list (list of int): A list of classes the module will determine the membership of
        - model (torch.nn.Module): The model to generate copies of used for determining membership
        """
        super(MembershipModel, self).__init__()
        self.class_set = class_list
        self.id = hash(str(self.class_set))
        self.model = model()


    def forward(self, x):
        """
        Overrides the default torch call
        
        Parameters:
        - x (torch.tensor): The input to run through the model
        
        Returns:
        - (torch.tensor): The result of running the input on the model
        """
        return self.model(x)


    def train_model(self, train_dataset, val_dataset=None, epochs=10, learning_rate=1e-3, class_threshold=0.5, cuda=True, verbose=True, logger=None):
        """
        Train the model

        Parameters:
        - train_dataset (np.array((np.array, int))): The dataset to train on which is a list of tuples (input, label)
        - val_dataset (np.array((np.array, int))): The dataset to validate on which is a list of tuples (input, label)
        - epochs (int): The number of epochs to train for
        - learning_rate (float): The learning rate of the model
        - class_threshold (float): The threshold above which we snap set membership values to 1 (in set)
        - cuda (bool): Whether training uses CUDA (default is True)
        - verbose (bool): Controls the amount of information printed
        - logger (Logger): The logger to log information to
        
        Returns:
        - epoch_losses (list of float): The loss history so we can more easily see when we finish!
        """
        loss_fn = torch.nn.MSELoss()
        
        best_model = {"loss": 999, "params": None}
        epoch_losses = []
        
        if logger:
            logger.info("Training model: {}".format(self.id))
        else:
            print("Training model: {}".format(self.id))
        
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

        self.to(device)
        opt = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)
        
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            self.train()
            losses = []
            batch_times = []

            for idx, data in enumerate(train_dataset):
                batch_start_time = time.time()
                
                inputs, labels = data
                labels = labels.type(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                out = self(inputs)
                out = torch.flatten(out) # Flatten outputs to a 1D tensor to align with labels
                loss = loss_fn(out, labels)
                loss.backward()

                opt.step()
                losses.append(loss.cpu().detach().numpy())
                
                del inputs, labels

                batch_times.append(time.time()-batch_start_time)

                if verbose:
                    if logger:
                        logger.info(progress_bar.progress_bar_with_eta(epoch, idx, len(train_dataset), 50, batch_times))
                    else:
                        print(progress_bar.progress_bar_with_eta(epoch, idx, len(train_dataset), 50, batch_times), end="\r")
                
                
            if val_dataset:
                self.eval()
                correct = 0
                total = 0
                val_losses = []
                for data in val_dataset:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    out = self(inputs)
                    out = torch.flatten(out)
                    v_loss = loss_fn(out, labels)
                    val_losses.append(v_loss.cpu().detach().numpy())
                    out = out.cpu().detach().numpy()
                    out = list(map(lambda x: 1.0 if x > class_threshold else 0.0, out))
                    for o,l in zip(out, labels):
                        if o == l:
                            correct += 1
                    total += labels.shape[0]
                    
                    del inputs, labels

            if verbose:
                if logger:
                    logger.info(progress_bar.progress_bar_with_eta(epoch, idx, len(train_dataset), 50, batch_times, val_acc=correct/total, 
                        epoch_time=time.time()-epoch_start_time,
                        epoch_loss=np.mean(losses)
                    ))
                else:
                    print(progress_bar.progress_bar_with_eta(epoch, idx, len(train_dataset), 50, batch_times, val_acc=correct/total, 
                        epoch_time=time.time()-epoch_start_time,
                        epoch_loss=np.mean(losses)
                    ))
            else:
                if logger:
                    logger.info("Epoch: {:03} - Loss: {:.3f} - Val loss: {:.4f} - Acc: {:.4f} ({:.3f}s)".format(epoch, np.mean(losses), np.mean(val_losses), correct/total, time.time()-epoch_start_time))
                print("Epoch: {:03} - Loss: {:.3f} - Acc: {:.4f} ({:.3f}s)".format(epoch, np.mean(losses), correct/total, time.time()-epoch_start_time))

            epoch_losses.append(np.mean(val_losses))
            # Store the model parameters if it performs better!
            # This would ideally be in validation?
            if np.mean(val_losses) < best_model['loss']:
                best_model['loss'] = np.mean(val_losses)
                best_model['params'] = copy.deepcopy(self.state_dict())
        
            scheduler.step(np.mean(val_losses))

        # Restore best model
        self.load_state_dict(best_model['params'])
        # Move model to CPU to free up GPU resources (otherwise we run into OOM errors with large models such as VGG)
        self.cpu()

        return epoch_losses


class ReedMullerAggregationNetwork(torch.nn.Module):
    """An Aggregation Network using Reed-Muller error correction"""

    def __init__(self, class_set, model, m, r, load_path=None, cuda=True):
        """
        Initialise an Aggregation Network with the specified class list
        
        Parameters:
        - class_set (list of int): The set of labels we intend to classify
        - model (torch.nn.Module): The basis model used to generate copies which we aggregate over
        - m (int): The number of variables in the polynomial
        - r (int): The highest degree of the polynomial
        - learning_rate (float): The learning rate of the networks
        - load_path (string): The path to load a pre-trained model from (default is None)
        - cuda (bool): Whether CUDA is used when training/testing the model (default is True)
        """
        super(ReedMullerAggregationNetwork, self).__init__()
        self.class_set = class_set
        self.num_networks = 2**m
        self.max_correctable_errors = np.floor((2**(m-r)-1)/2)
        
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

        if not load_path:
            # Generate the sets each network is trained to determine membership of
            # and the class bitstrings for classification
            self.class_permutations, self.bitstring_classes = rmc.generate_reed_muller_sets(m, self.class_set)
            self.num_networks = len(self.class_permutations)

            # Check the Hamming distances (out of interest)
            ham.analyse_sets(self.bitstring_classes)
            
            # Generate the networks
            self.network_list = self.generate_networks(model)
        else:
            # Load networks from specified path
            self.load_from_saved(load_path, model)


    def generate_networks(self, model):
        """
        Generate the models ready for training
        
        Parameters:
        - model (torch.nn.Module): The model used to determine set membership
        
        Returns:
        - (torch.nn.ModuleList): The networks assigned as submodules collected in a list
        """
        return torch.nn.ModuleList([MembershipModel(ls, model) for ls in self.class_permutations])


    def to_device(self, device):
        """
        Move all contained models to a new device

        Parameters:
        - device (str): The device to move to
        """
        self.device = device
        [m.to(self.device) for m in self.network_list]
    

    def forward(self, x):
        """
        Forward pass of the achitecture

        Parameters:
        - x (torch.tensor): The input data to pass to the model

        Returns:
        - membership_vector (torch.tensor): Returns the raw combined output from each network for each input
        """
        # [m.to(self.device) for m in self.network_list]
        membership_vector = torch.cat([torch.unsqueeze(m(x), dim=0) for m in self.network_list])
        # # This code is useful if we have a large model with limited GPU memory available to us
        # # but it slows down evaluation massively!
        # membership_vector = []
        # for m in self.network_list:
        #     m.to('cuda')
        #     # Using m(x).detach().cpu() to avoid OOM errors when using larger models e.g. VGG
        #     membership_vector.append(torch.unsqueeze(m(x).detach().cpu(), dim=0))
        #     m.cpu()
        # membership_vector = torch.cat(membership_vector)
        membership_vector = torch.transpose(membership_vector, 0, 1)
        membership_vector = torch.squeeze(membership_vector)
        # [m.to('cpu') for m in self.network_list]
        return membership_vector


    def thresholded_forward(self, x, class_threshold=0.5, error_correction=False, num_bits=1):
        """
        Use the aggregated model to make predictions with a classification threshold

        Parameters:
        - x (torch.tensor): The input to run through the aggregated model
        - class_threshold (float): The threshold a result must exceed to be classified as a 1 bit (default is 0.5)
        - error_correction (bool): Whether or not error correction should be applied (if false all following parameters are ignored) (default is false)
        - num_bits (int): The number of bits that can be corrected (provided error_correction=True) (default is 1)
        
        Returns:
        - (np.array): The result of running the input on the aggregated model
        """
        # Get the raw results from the models predicting set membership of the inputs
        membership_vector = self.forward(x).cpu().detach().numpy()
        
        # Handle bug which occurs when x is a single element
        if x.shape[0] == 1:
            membership_vector = np.expand_dims(membership_vector, axis=0)

        # Threshold all of the values
        threshold = np.vectorize(lambda z: 1 if z > class_threshold else 0)
        membership_vector = threshold(membership_vector)

        # Extract the classification (if any) from the membership vector
        res = []
        for vec in membership_vector:
            # Convert the vector to a bitstring
            predicted_bitstr = ''.join([str(int(v)) for v in vec])

            # Find a class if one exists
            predicted_class = np.nan
            for c, class_bitstr in enumerate(self.bitstring_classes):
                if predicted_bitstr == class_bitstr:
                    predicted_class = c
                    break

            # If we have no definitive class and error correction is active, try error correction!
            if np.isnan(predicted_class) and error_correction:
                
                hamming_distances = []

                for c, class_bitstr in enumerate(self.bitstring_classes):
                    h_dist = ham.hamming_distance(class_bitstr, predicted_bitstr)
                    hamming_distances.append((c, h_dist))

                # Sort Hamming distances in ascending order
                hamming_distances.sort(key=lambda x: x[1])
                
                # Take the class associated with the smallest Hamming distance
                p_class, h_dist = hamming_distances[0]
                
                # Check for ambiguity with any other classes (other classes with the same Hamming distance) and that we're below or equal to the num_bits threshold
                _, h_dist2 = hamming_distances[1]
                
                if h_dist <= num_bits and not(h_dist == h_dist2):
                    predicted_class = p_class

            res.append(predicted_class)

        return np.asarray(res)


    def evaluate(self, dataset, batch_size=64, class_threshold=0.5, error_correction=False, num_bits=1):
        """
        Evaluate the model on a dataset

        Parameters:
        - dataset (list of (numpy.array, int)): The dataset to evaluate the model on
        - batch_size (int): The size of the batch (default is 64)
        - class_threshold (float): The threshold a result must exceed to be classified as a 1 bit (default it 0.5)
        - error_correction (bool): Whether or not error correction should be applied (if false all following parameters are ignored) (default is false)
        - num_bits (int): The number of bits that can be corrected (provided error_correction=True) (default is 1)
        
        Returns:
        - (dict): A dictionary of results
            - "correct" (float): The ratio of correctly classified items
            - "reject" (float): The ratio of rejected items
            - "incorrect" (float): The ratio of incorrect items
        """
        
        if error_correction:
            assert num_bits <= self.max_correctable_errors, "Amount of requested error correction exceeds the availability!"

        # Create the dataset
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.to_device(self.device)
        [m.eval() for m in self.network_list]
        
        correct = 0
        rejected = 0
        total = 0
        
        for inputs, labels in dataset:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            predictions = self.thresholded_forward(inputs, class_threshold=class_threshold, error_correction=error_correction, num_bits=num_bits)
            total += labels.shape[0]
            
            for p, l in zip(predictions, labels):
                correct += 1 if p == l else 0

            rejected += len(list(filter(lambda x: np.isnan(x), predictions)))

        incorrect = total-correct-rejected
        self.to_device('cpu')

        return {
            "correct": correct/total,
            "rejected": rejected/total,
            "incorrect": incorrect/total,
        }


    def train_model(self, data, val_data, epochs, batch_size=64, learning_rate=1e-3, class_threshold=0.5, verbose=True, logger=None):
        """
        Train the model

        Parameters:
        - data (list of (numpy.array, int)): The training dataset which is a list of tuples, containing the input element and the class
        - val_data (list of (numpy.array, int)): The validation dataset
        - epochs (int): The number of epochs to train for
        - batch_size (int): The size of the batches used to train/validate (default is 64)
        - learning_rate (float): The learning rate of the models (default is 1e-3)
        - class_threshold (float): The threshold a result must exceed to be classified as a 1 bit (default it 0.5)
        - verbose (bool): Whether lots of information should be printed during training (default is True)
        - logger (Logger): A logger to print the information to

        Returns:
        - model_loss_histories (list of list of float): The historical losses for each model
        """
        
        self.to_device(self.device)
        model_loss_histories = []
        count = 1
        for model in self.network_list:
            if logger:
                logger.info("Training model {}: {} - {}/{}".format(model.id, model.class_set, count, len(self.network_list)))
            else:
                print("Training model {}: {} - {}/{}".format(model.id, model.class_set, count, len(self.network_list)))

            model_dataset = []
            classes = model.class_set
            for i,l in data:
                if l in classes:
                    model_dataset.append((i, 1.0))
                else:
                    model_dataset.append((i, 0.0))
            model_dataset = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, shuffle=True)
            
            # Format validation data
            val_dataset = []
            for i,l in val_data:
                if l in classes:
                    val_dataset.append((i, 1.0))
                else:
                    val_dataset.append((i, 0.0))
            validation_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            
            cuda = False
            if self.device == "cuda":
                cuda = True

            loss_history = model.train_model(model_dataset, validation_dataset, epochs=epochs, class_threshold=class_threshold, cuda=cuda, learning_rate=learning_rate, verbose=verbose, logger=logger)
            model_loss_histories.append(loss_history)

            count += 1
        self.to_device(self.device)
        return model_loss_histories


    def save_aggnet(self, path):
        """
        Save the RMAggNet data for later re-use

        Parameters:
        - path (string): The path to save the data to
        """
        if not os.path.isdir(path):
            print("Directory {} does not exist - creating...".format(path))
            os.makedirs(path)

        model_dict = {}
        model_dict['bitstring_classes'] = self.bitstring_classes
        model_dict['num_networks'] = self.num_networks
        model_dict['max_correctable_errors'] = self.max_correctable_errors
        nets = {}
        for model in self.network_list:
            nets[str(model.id)] = model.class_set
            torch.save(model.state_dict(), os.path.join(path, str(model.id)))
        model_dict['trained_networks'] = nets
        
        with open(os.path.join(path, "agg_model.p"), "wb") as sfile:
            pickle.dump(model_dict, sfile)
    

    def load_from_saved(self, path, model):
        """
        Throw away the created sets and load the model from saved data

        Parameters:
        - path (string): The path to load the data from
        - model (torch.nn.Module): The basis model used to generate copies which we aggregate over
        """
        with open(os.path.join(path, "agg_model.p"), "rb") as pfile:
            model_dict = pickle.load(pfile)
        model_list = []
        for m in model_dict['trained_networks']:
            model_id = m
            model_set = model_dict['trained_networks'][m]
            classifier = MembershipModel(model_set, model)
            classifier.load_state_dict(torch.load(os.path.join(path, str(model_id))))
            model_list.append(classifier)

        self.network_list = torch.nn.ModuleList(model_list)
        
        self.bitstring_classes = model_dict['bitstring_classes']
        self.num_networks = len(self.network_list)
        self.max_correctable_errors = model_dict['max_correctable_errors']

def aggnet_eval(rm_aggnet, dataset, max_correction, batch_size=64, thresholds=[0.5], logger=None, ood=False):
    """
    Evaluate RMAggNet over a dataset, varying the amount of correction and the thresholds

    Parameters:
    - rm_aggnet (torch.nn.Module): The RMAggNet torch model
    - dataset (list of (numpy.array, int)): The dataset to evaluate on which is a list of tuples, containing the input element and the class
    - max_correction (int): The maximum number of bits to correct
    - batch_size (int, optional): The size of the batch of data (default is 64)
    - thresholds (list of float, optional): A list of thersholds to iterate over (default is [0.5])
    - logger (Logger, optional): A logger to print the information to (default is None)
    - ood (bool, optional): Whether the data is naturally out-of-distribution (default is False)
    """
    logger.info("No error correction")
    for t in thresholds:
        res = rm_aggnet.evaluate(dataset, batch_size=batch_size, class_threshold=t)
        if ood:
            logger.info("| {:.1f} | {:.2f} | {:.2f} |".format(
                t, res['rejected']*100, (res['incorrect']+res['correct'])*100 
            ))
        else:
            logger.info("| {:.1f} | {:.2f} | {:.2f} | {:.2f} |".format(
                t, res['correct']*100, res['rejected']*100, res['incorrect']*100 
            ))

    for num_corrections in range(1, max_correction+1):
        logger.info("EC={}".format(num_corrections))
        for t in thresholds:
            res = rm_aggnet.evaluate(dataset, batch_size=batch_size, class_threshold=t, error_correction=True, num_bits=num_corrections)
            if ood:
                logger.info("| {:.1f} | {:.2f} | {:.2f} |".format(
                    t, res['rejected']*100, (res['incorrect']+res['correct'])*100
                ))
            else:
                logger.info("| {:.1f} | {:.2f} | {:.2f} | {:.2f} |".format(
                    t, res['correct']*100, res['rejected']*100, res['incorrect']*100 
                ))
