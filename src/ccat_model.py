import torch
import numpy as np
import math

# CCAT imports
import sys
sys.path.insert(1, "CCAT") # Add the CCAT directory as an import
import common
import attacks


class CCAT:

    def __init__(self, model):
        self.ccat_model = model

    def adversarial_training(self, train_loader, test_loader, epochs, optimizer, epsilon, cuda=True, save_path=None):
        """
        Adversarially train the CCAT model

        Parameters:
        - train_loader (DataLoader): The training data to train on (and create adversarial examples to train on from)
        - test_loader (DataLoader): The testing data to test the resulting model on
        - epochs (int): The number of epochs to train for
        - optimizer (torch.optim.Optimizer): The training optimizer
        - epsilon (float): The strength of the adversarial perturbation
        - cuda (bool, optional): Enable CUDA for training (default is True)
        - save_path (string, optional): The path to save the resulting model to (default is None)
        """
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.ccat_model.to(device)
        self.ccat_model.train()

        batches_per_epoch = len(train_loader)
        gamma = 0.97
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])

        batches_per_epoch = len(train_loader)
        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 40
        attack.base_lr = 0.005
        attack.momentum = 0.9
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()
        objective = attacks.objectives.UntargetedF7PObjective()

        loss = common.torch.cross_entropy_divergence
        transition = common.utils.partial(common.torch.power_transition, norm=attacks.norms.LInfNorm(), gamma=12, epsilon=0.3)
        trainer = common.train.ConfidenceCalibratedAdversarialTraining(
            self.ccat_model, 
            train_loader,
            # It may seem a bit strange to include the test_loader while training, but it isn't acutually used unless
            # we call trainer.test() so it doesn't really do anything here. 
            test_loader, 
            optimizer, 
            scheduler, 
            attack, 
            objective, 
            loss, 
            transition, 
            fraction=0.5, 
            cuda=cuda, 
        )

        for epoch in range(epochs):
            trainer.step(epoch)
        
        if save_path:
            torch.save(self.ccat_model.state_dict(), save_path)

    def evaluate(self, data, confidence_thresholds, cuda=True, ood=False, logger=None):
        """
        Evaluate the CCAT model over the specified confidence thresholds

        Parameters:
        - data (DataLoader): The dataset to evaluate on
        - confidence_thresholds (list of float): A list of confidence thresholds to evaluate over
        - cuda (bool, optional): Enable CUDA for evaluation (default is True)
        - ood (bool, optional): Whether the data is considered out of the natural distribution (this does not apply to adversaries) (default is False)
        - logger (Logger, optional): A logger to print the results to (default is None)
        """
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.ccat_model.to(device)
        self.ccat_model.eval()
        
        confidences = None
        results = []
        labels = []

        for inputs, targets in data:

            inputs = common.torch.as_variable(inputs, cuda)
            targets = common.torch.as_variable(targets, cuda)

            outputs = self.ccat_model(inputs)
            results.append(outputs.detach().cpu().numpy())
            labels.append(targets.cpu().numpy())
            confidences = common.numpy.concatenate(
                confidences, 
                torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy()
            )
        
        results = np.concatenate(results)
        labels = np.concatenate(labels)
        
        for t in confidence_thresholds:
            correct = 0
            rejected = 0
            for result, label, conf in zip(np.argmax(results, axis=1), labels, confidences):
                if conf >= t:
                    correct += 1 if result == label else 0
                else:
                    rejected += 1
            
            incorrect = results.shape[0]-rejected
            if logger:
                if not ood:
                    logger.info("{:.1f} | {:.2f} | {:.2f} | {:.2f}".format(t, (correct/results.shape[0])*100, (rejected/results.shape[0])*100, (1-((correct+rejected)/results.shape[0]))*100))
                else:
                    logger.info("{:.1f} | {:.2f} | {:.2f}".format(t, (rejected/results.shape[0])*100, incorrect/results.shape[0]*100))
            else:
                if not ood:
                    print("{:.1f} | {:.2f} | {:.2f} | {:.2f}".format(t, correct/results.shape[0], rejected/results.shape[0], 1-((correct+rejected)/results.shape[0])))
                else:
                    print("{:.1f} | {:.2f} | {:.2f} ".format(t, rejected/results.shape[0], incorrect/results.shape[0]))

        self.ccat_model.to('cpu')