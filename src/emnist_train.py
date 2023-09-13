import torch
import torchvision
import datetime
import logging

from rm_aggnet import ReedMullerAggregationNetwork as RMAggNet 
import ensemble
from ccat_model import CCAT

from utils.dataset_transform import transform_mnist_dataset

# Import models
from arch.rm_models import RMEMNISTModel as RMModel
from arch.ensemble_models import EnsEMNISTModel as EnsModel
from arch.standard_models import EMNISTModel as StandardModel 

def train_emnist(models, save_dir, cuda):
    """
    Train specified models on the EMNIST dataset

    Parameters:
    - models (list of string): A list of models to train (choose from 'rmaggnet', 'ensemble', 'ccat' and 'surrogate')
    - save_dir (string): The directory to save the models to
    - cuda (bool): Whether to run with CUDA
    """
    #### Set up the logger
    logname = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")

    #logging.basicConfig(level=logging.INFO, format="(%asctime)s - %(levelname)s - %(message)s", filename="{}_emnist.log".format(logname))
    logging.basicConfig(level=logging.INFO, filename="{}_emnist.log".format(logname))
    prog_logger = logging.getLogger(__name__)
    ####

    batch_size = 256
    epochs = 200

    # Load EMNIST (131,600 samples with 47 balanced classes)
    # www.westernsydney.edu.au/__data/assets/text_file/0019/1204408/EMNIST_Readme.txt
    train_mnist = torchvision.datasets.EMNIST('/tmp', split='balanced', train=True, download=True)
    test_mnist = torchvision.datasets.EMNIST('/tmp', split='balanced', train=False, download=True)

    train_dataset_full = transform_mnist_dataset(train_mnist)
    val_ratio = 0.2
    train_dataset = train_dataset_full[:int(len(train_dataset_full)-len(train_dataset_full)*val_ratio)]
    validation_dataset = train_dataset_full[int(len(train_dataset_full)-len(train_dataset_full)*val_ratio):]
    test_dataset = transform_mnist_dataset(test_mnist)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ################################# RMAggNet
    if 'rmaggnet' in models:
        rm_aggnet = RMAggNet([x for x in range(47)], RMModel, m=5, r=1, cuda=cuda)
        model_loss_hist = rm_aggnet.train_model(train_dataset, validation_dataset, batch_size=batch_size, epochs=epochs, learning_rate=1e-4, class_threshold=0.5, verbose=False, logger=prog_logger)
        rm_aggnet.save_aggnet("{}/rmaggnet_emnist".format(save_dir))
        prog_logger.info("===RMAggNet===")
        rm_aggnet.aggnet_eval(rm_aggnet, test_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=7, logger=prog_logger)

        del rm_aggnet


    ################################# Ensemble
    if 'ensemble' in models:
        ensemble_model = ensemble.Ensemble(EnsModel, 32)
        ensemble_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-4, verbose=False, logger=prog_logger, cuda=cuda)
        ensemble_model.save_ensemble("{}/ens_emnist".format(save_dir))

        prog_logger.info("===Ensemble===")
        ensemble.ensemble_eval(ensemble_model, test_loader, thresholds=[x/10 for x in range(11)], cuda=cuda)

        del ensemble_model


    ################################# CCAT (may take ~17 hours)
    if 'ccat' in models:
        ccat_epochs = 50
        # Setup a model, the optimizer, learning rate scheduler.
        ccat_base_model = EnsModel()
        ccat = CCAT(ccat_base_model)
        optimizer = torch.optim.SGD(ccat.ccat_model.parameters(), lr=0.01, momentum=0.9)


        ccat.adversarial_training(
            train_loader, 
            test_loader, 
            epochs=ccat_epochs, 
            optimizer=optimizer, 
            epsilon=0.3, 
            cuda=cuda, 
            save_path="{}/ccat_emnist.pth.tar".format(save_dir)
        )

        print("=== CCAT ===")
        ccat.evaluate(test_loader, confidence_thresholds=[x/10 for x in range(11)], cuda=cuda, logger=prog_logger)

        del ccat

    ################################# Standard (used for closed-box transfer attacks)
    if 'surrogate' in models:
        standard_model = StandardModel()
        standard_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-4, verbose=False, logger=prog_logger, cuda=cuda)
        standard_model.evaluate(test_loader, logger=prog_logger, cuda=cuda)
        torch.save(standard_model.state_dict(), "{}/standard_emnist.pth.tar".format(save_dir))

if __name__ == "__main__":
    train_emnist(["rmaggnet", "ensemble", "ccat", "surrogate"], save_dir="trained_models", cuda=torch.cuda.is_available())