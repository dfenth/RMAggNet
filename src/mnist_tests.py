import torch
import torchvision
import datetime
import logging

import rm_aggnet
from rm_aggnet import ReedMullerAggregationNetwork as RMAggNet 
import ensemble
from ccat_model import CCAT

from utils.dataset_transform import transform_mnist_dataset

# Import models
from arch.rm_models import RMMNISTModel as RMModel
from arch.ensemble_models import EnsMNISTModel as EnsModel
from arch.standard_models import MNISTModel as StandardModel 


#### Set up the logger
logname = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")

#logging.basicConfig(level=logging.INFO, format="(%asctime)s - %(levelname)s - %(message)s", filename="{}_emnist.log".format(logname))
logging.basicConfig(level=logging.INFO, filename="{}_mnist.log".format(logname))
prog_logger = logging.getLogger(__name__)
####

batch_size = 64
epochs = 50
ccat_epochs = 50

# Load MNIST
train_mnist = torchvision.datasets.MNIST('/tmp', train=True, download=True)
test_mnist = torchvision.datasets.MNIST('/tmp', train=False, download=True)

train_dataset_full = transform_mnist_dataset(train_mnist)
val_ratio = 0.2
train_dataset = train_dataset_full[:int(len(train_dataset_full)-len(train_dataset_full)*val_ratio)]
validation_dataset = train_dataset_full[int(len(train_dataset_full)-len(train_dataset_full)*val_ratio):]
test_dataset = transform_mnist_dataset(test_mnist)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


################################# RMAggNet
rm_aggnet = RMAggNet([x for x in range(10)], RMModel, m=4, r=1, learning_rate=1e-4)
model_loss_hist = rm_aggnet.train_model(train_dataset, validation_dataset, batch_size=batch_size, epochs=epochs, learning_rate=1e-4, class_threshold=0.5, verbose=False, logger=prog_logger)
rm_aggnet.save_aggnet("trained_models/rmaggnet_mnist")
prog_logger.info("===RMAggNet===")
rm_aggnet.aggnet_eval(rm_aggnet, test_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=prog_logger)

del rm_aggnet

################################# Ensemble
ensemble_model = ensemble.Ensemble(EnsModel, 16)
ensemble_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-4, verbose=False, logger=prog_logger)
ensemble_model.save_ensemble("trained_models/ens_mnist")

prog_logger.info("===Ensemble===")
ensemble.ensemble_eval(ensemble_model, test_loader, thresholds=[x/10 for x in range(11)])

del ensemble_model

################################# CCAT
# Setup a model, the optimizer, learning rate scheduler.
ccat_base_model = EnsModel()
ccat = CCAT(ccat_base_model)
optimizer = torch.optim.SGD(ccat.ccat_model.parameters(), lr=0.1, momentum=0.9)

ccat.adversarial_training(
    train_loader, 
    test_loader, 
    epochs=epochs, 
    optimizer=optimizer, 
    epsilon=0.3, 
    cuda=True, 
    save_path='trained_models/ccat_mnist.pth.tar'
)

print("=== CCAT ===")
ccat.evaluate(test_loader, confidence_thresholds=[x/10 for x in range(11)], cuda=True, logger=prog_logger)

del ccat

################################# Standard (used for closed-box transfer attacks)
standard_model = StandardModel()
standard_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-4, verbose=False, logger=prog_logger)
standard_model.evaluate(test_loader, logger=prog_logger)
torch.save(standard_model.state_dict(), 'trained_models/standard_mnist.pth.tar')
