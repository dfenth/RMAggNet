import torch
import numpy as np
import time
import os
import pickle

import utils.progress_bar as progress_bar

class Ensemble(torch.nn.Module):

    def __init__(self, model, num_ensemble, load_path=None):
        """
        Initialise the Ensemble class

        Parameters:
        - model (torch.nn.Module): The mode which will be used for the ensemble
        - num_ensemble (int): The number models making the ensemble
        - load_path (string, optional): The path to load a model from (default is None)
        """
        super(Ensemble, self).__init__()
        if not load_path:
            self.ref_model = model    
            self.ensemble = [self.ref_model() for _ in range(num_ensemble)]
            self.logit_weights = torch.nn.Linear(num_ensemble, 1, bias=False) # A bit of a hack to optimise the logit weights
            # Ensure all logit weights sum to 1
            self.logit_weights.weight.data = torch.nn.functional.softmax(self.logit_weights.weight, dim=1)
            
            # Force all logit weights to be the same
            new_weights = torch.ones(size=self.logit_weights.weight.shape)/num_ensemble
            self.logit_weights.weight.data = new_weights

        else:
            self.load_from_saved(load_path, model)
            
            # Force all logit weights to be the same (necessary here for any old saved models where this was not standardised!)
            new_weights = torch.ones(size=self.logit_weights.weight.shape)/num_ensemble
            self.logit_weights.weight.data = new_weights


    def train_on_data(self, training_data, validation_data, epochs=10, lr=1e-3, cuda=True, verbose=True, logger=None):
        """
        Train on the data

        Parameters:
        - training_data (DataLoader): The training data as a torch data loader
        - validation_data (DataLoader): The validation data as a torch data loader
        - epochs (int, optional): The number of epochs to train for (default is 10)
        - lr (float, optional): The learning rate of the model (default is 1e-3)
        - cuda (bool, optional): Whether CUDA is used for training (default is True)
        - verbose (bool, optional): The verbosity of messages (default is True)
        - logger (Logger, optiona): The logger to log progress messages to (default is None)
        """
        if logger:
            logger.info("Logit weights: {}".format(self.logit_weights))
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        
        total_training_time = time.time()
        self.logit_weights.to(device)

        for net_id, net in enumerate(self.ensemble):
            if logger:
                logger.info("Training network {}".format(net_id+1))
            else:
                print("Training network {}".format(net_id+1))
            net.to(device)
            
            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                
                net.train()
                # Set up optimiser and loss function for the network
                opt = torch.optim.Adam(params=net.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)
                loss_fn = torch.nn.CrossEntropyLoss()
                train_losses = []
                batch_times = []
                train_correct = 0
                train_total = 0
                
                for batch_id, data in enumerate(training_data):
                    batch_start_time = time.time()
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    opt.zero_grad()
                    out = net.forward(inputs)
                    predictions = torch.nn.functional.softmax(out, dim=1)
                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    opt.step()
                    
                    train_losses.append(loss.to('cpu').detach().numpy())
                    
                    train_total += labels.shape[0]
                    train_correct += int((predictions.argmax(dim=1) == labels).sum())

                    batch_times.append(time.time()-batch_start_time)
                    if verbose:
                        print(progress_bar.progress_bar_with_eta(epoch, batch_id, len(training_data), 50, batch_times), end='\r')
                
                # Validation
                net.eval()
                val_losses = []
                total = 0
                correct = 0

                for inputs, labels in validation_data:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    out = net.forward(inputs)
                    predictions = torch.nn.functional.softmax(out, dim=1)
                    loss = loss_fn(predictions, labels)
                    val_losses.append(loss.to('cpu').detach().numpy())
                    total += labels.shape[0]
                    correct += int((predictions.argmax(dim=1) == labels).sum())

                
                if verbose:
                    if logger:
                        logger.info(progress_bar.progress_bar_with_eta(
                            epoch, 
                            batch_id, 
                            len(training_data), 
                            50, 
                            batch_times, 
                            correct/total, 
                            time.time()-epoch_start_time, 
                            np.mean(val_losses))
                        )
                    else:
                        print(progress_bar.progress_bar_with_eta(
                            epoch, 
                            batch_id, 
                            len(training_data), 
                            50, 
                            batch_times, 
                            correct/total, 
                            time.time()-epoch_start_time, 
                            np.mean(val_losses))
                        )
                else:
                    if logger:
                        logger.info("Epoch: {:03} - Train Loss: {:.3f} Train Acc: {:.3f} -- Val Loss: {:.3f} - Val Acc: {:.4f} ({:.3f}s)".format(
                            epoch, np.mean(train_losses), train_correct/train_total, np.mean(val_losses), correct/total, time.time()-epoch_start_time)
                        )
                    else:
                        print("Epoch: {:03} - Loss: {:.3f} - Acc: {:.4f} ({:.3f}s)".format(epoch, np.mean(val_losses), correct/total, time.time()-epoch_start_time))
                
                scheduler.step(np.mean(val_losses))

            # Unload the network from the GPU
            net.to('cpu')

        if logger:
            logger.info("Training complete in: {:.2f}s".format(time.time()-total_training_time))
        else:
            print("Training complete in: {:.2f}s".format(time.time()-total_training_time))
    
    
    def evaluate_with_reject(self, dataset, agreement, cuda=True):
        """
        Evaluate the performance with a reject option

        Parameters:
        - dataset (DataLoader): The evaluation data as a torch data loader
        - agreement (float): The percentage of networks that must agree on a class for a valid result
        - cuda (bool, optional): Whether CUDA is used for evaluation (default is True)

        Returns:
        - (dict): A dictionary of results
            - "correct" (float): The ratio of correctly classified items
            - "reject" (float): The ratio of rejected items
            - "incorrect" (float): The ratio of incorrect items
        """

        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        
        self.logit_weights.to(device)
        # Outputs organised as net, elem, result
        outputs = []
        add_labels = True # Hacky label things because we need to extract them from the dataset iterable!
        all_labels = []

        for net in self.ensemble:
            net.eval()
            net.to(device)
            net_output = []
            for inputs, labels in dataset:
                inputs = inputs.to(device)
                
                out = net.forward(inputs)
                net_output.append(out.cpu().detach().numpy()) # Add the network output to the list of outputs

                if add_labels:
                    all_labels.append(labels.cpu())
            
            add_labels = False
            net_output = np.concatenate(net_output)
            # Expand net_output so
            outputs.append(net_output)

            net.to('cpu')

        outputs = np.stack(outputs) 
        # Reshape to be elem, net, result
        outputs = np.transpose(outputs, axes=[1,0,2])
        
        # Ensemble the outputs from the networks
        ensemble_output = []
        for elem in outputs:
            # Extract the index of the maximum value (this is the prediction)
            pred_classes = np.argmax(elem, axis=1)
            # Count the number of predictions per class
            votes = np.bincount(pred_classes, minlength=10)

            # Check if the highest number of votes exceeds the threshold
            if np.max(votes) >= int(len(self.ensemble)*agreement): # int gives the floor here!
                ensemble_output.append(np.argmax(votes))
            else:
                ensemble_output.append(np.nan)

        ensemble_output = np.asarray(ensemble_output)

        # Compare output to labels
        correct = 0
        rejected = 0
        incorrect = 0
        all_labels = np.concatenate(all_labels)
        for o, l in zip(ensemble_output, all_labels):
            if np.isnan(o):
                rejected += 1
            elif o == l:
                correct += 1
            else:
                incorrect += 1
        
        return {
                'correct': correct/all_labels.shape[0], 
                'rejected': rejected/all_labels.shape[0], 
                'incorrect':incorrect/all_labels.shape[0]
                }


    def forward(self, x):
        """
        Use forward to produce the adversaries using ensemble in logits method from https://arxiv.org/pdf/1710.06081.pdf

        Parameters:
        - x (torch.tensor): The input to the model

        Returns:
        - output (torch.tensor): The weighted output
        """
                
        output = torch.cat([torch.unsqueeze(net(x), dim=0) for net in self.ensemble]) # Produces a tensor of shape num_nets x num_samples x num_classes
        output = torch.transpose(output, 0, 1) # Transpose so we have sample x network x classes
        # Weight each of the network's class predictions and sum the predictions per class
        # vmap transpose leads to class x net matrix
        # Then multiplication with logit_weights (net x 1 matrix) results in class x 1 which is the weighted class prediction
        mul_weights = torch.vmap(lambda x: self.logit_weights(torch.transpose(x, 0, 1)))
        output = mul_weights(output)
        
        # Re-squeeze!
        output = torch.squeeze(output)

        return output # NOTE: These are still logits!
    

    def save_ensemble(self, path):
        """
        Save the Ensemble network to file

        Parameters:
        - path (string): The path to save the model to
        """
        if not os.path.isdir(path):
            print("Directory {} does not exist - creating...".format(path))
            os.makedirs(path)

        model_dict = {}
        model_dict['models'] = []
        for idx, model in enumerate(self.ensemble):
            torch.save(model.state_dict(), os.path.join(path, str(idx)))
            model_dict['models'].append(idx)

        torch.save(self.logit_weights.state_dict(), os.path.join(path, "logit_weights"))
        with open(os.path.join(path, "ens_model.p"), "wb") as sfile:
            pickle.dump(model_dict, sfile)
    

    def load_from_saved(self, path, model):
        """
        Load ensemble model from saved data
        
        Parameters:
        - path (string): The path to load the model data from
        - model (torch.nn.Module): The model to pass the state_dict to
        """
        self.ensemble = []
        model_dict = pickle.load(open(os.path.join(path, "ens_model.p"), "rb"))
        model_list = []
        for m_id in model_dict['models']:
            classifier = model()
            classifier.load_state_dict(torch.load(os.path.join(path, str(m_id))))
            self.ensemble.append(classifier)

        self.logit_weights = torch.nn.Linear(len(self.ensemble), 1, bias=False)
        self.logit_weights.load_state_dict(torch.load(os.path.join(path, "logit_weights")))


def ensemble_eval(ensemble, dataset, thresholds, logger=None, ood=False):
    """
    Evaluate Ensemble over a dataset, varying the thresholds

    Parameters:
    - ensemble (torch.nn.Module): The Ensemble torch model
    - dataset (DataLoader): The dataset to evaluate on
    - thresholds (list of float): A list of thersholds to iterate over
    - logger (Logger, optional): A logger to print the information to (default is None)
    - ood (bool, optional): Whether the data is naturally out-of-distribution (default is False)
    """
    for t in thresholds:
        res = ensemble.evaluate_with_reject(dataset, t)
        if ood:
            logger.info("{:.1f} | {:.2f} | {:.2f}".format(
                t, res['rejected']*100, (res['incorrect']+res['correct'])*100 
            ))
        else:
            logger.info("{:.1f} | {:.2f} | {:.2f} | {:.2f}".format(
                t, res['correct']*100, res['rejected']*100, res['incorrect']*100 
            ))