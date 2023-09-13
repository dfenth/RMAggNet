import torch
import torchvision
import time
import datetime
import logging

from rm_aggnet import ReedMullerAggregationNetwork as RMAggNet 
import ensemble
from ccat_model import CCAT

from utils.dataset_transform import transform_cifar_dataset

# Import models
from arch.rm_models import RMCIFARModel as RMModel
from arch.ensemble_models import EnsCIFARModel as EnsModel
from arch.standard_models import CIFARModel as StandardModel 


def train_cifar(models, save_dir, cuda):
    """
    Train specified models on the CIFAR-10 dataset

    Parameters:
    - models (list of string): A list of models to train (choose from 'rmaggnet', 'ensemble', 'ccat' and 'surrogate')
    - save_dir (string): The directory to save the models to
    - cuda (bool): Whether to run with CUDA
    """
    #### Set up the logger
    logname = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")

    #logging.basicConfig(level=logging.INFO, format="(%asctime)s - %(levelname)s - %(message)s", filename="{}_emnist.log".format(logname))
    logging.basicConfig(level=logging.INFO, filename="{}_cifar.log".format(logname))
    prog_logger = logging.getLogger(__name__)
    ####

    batch_size = 256
    epochs = 500

    # Load the CIFAR dataset
    train_ds = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10('/tmp', train=False, download=True)

    train_dataset_full = transform_cifar_dataset(train_ds)
    val_ratio = 0.1
    train_dataset = train_dataset_full[:int(len(train_dataset_full)-len(train_dataset_full)*val_ratio)]
    validation_dataset = train_dataset_full[int(len(train_dataset_full)-len(train_dataset_full)*val_ratio):]
    test_dataset = transform_cifar_dataset(test_ds)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ################################# RMAggNet
    if 'rmaggnet' in models:
        rm_start_time = time.time()
        rm_aggnet = RMAggNet([x for x in range(10)], RMModel, m=4, r=1, cuda=cuda)
        model_loss_hist = rm_aggnet.train_model(train_dataset, validation_dataset, batch_size=batch_size, epochs=epochs, class_threshold=0.5, verbose=False, logger=prog_logger)
        rm_aggnet.save_aggnet("{}/rmaggnet_cifar".format(save_dir))
        
        prog_logger.info("===RMAggNet===")
        rm_aggnet.aggnet_eval(rm_aggnet, test_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=prog_logger)
        prog_logger.info("RMAggNet training took: {:.2f}s".format(time.time()-rm_start_time))

        del rm_aggnet


    ################################# Ensemble
    if 'ensemble' in models:
        ens_start_time = time.time()
        ensemble_model = ensemble.Ensemble(EnsModel, 16)
        ensemble_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-3, verbose=False, logger=prog_logger, cuda=cuda)
        ensemble_model.save_ensemble("{}/ens_cifar".format(save_dir))

        prog_logger.info("===Ensemble===")
        ensemble.ensemble_eval(ensemble_model, test_loader, thresholds=[x/10 for x in range(11)], cuda=cuda)

        prog_logger.info("Ensemble training took: {:.2f}s".format(time.time()-ens_start_time))

        del ensemble_model


    ################################# CCAT
    if 'ccat' in models:
        ccat_start_time = time.time()
        ccat_epochs = 400

        # Setup a model, the optimizer, learning rate scheduler.
        ccat_base_model = EnsModel()
        ccat = CCAT(ccat_base_model)
        optimizer = torch.optim.Adam(ccat.ccat_model.parameters(), lr=1e-3)

        ccat.adversarial_training(
            train_loader, 
            test_loader, 
            epochs=ccat_epochs, 
            optimizer=optimizer, 
            epsilon=0.3, 
            cuda=cuda, 
            save_path="{}/ccat_cifar.pth.tar".format(save_dir)
        )

        print("=== CCAT ===")
        ccat.evaluate(test_loader, confidence_thresholds=[x/10 for x in range(11)], cuda=cuda, logger=prog_logger)
        prog_logger.info("CCAT training took: {:.2f}s".format(time.time()-ccat_start_time))

        del ccat

    ################################# Standard (used for closed-box transfer attacks)
    if 'surrogate' in models:
        standard_model = StandardModel()
        standard_model.train_on_data(train_loader, val_loader, epochs=epochs, lr=1e-3, verbose=False, logger=prog_logger, cuda=cuda)
        standard_model.evaluate(test_loader, logger=prog_logger, cuda=cuda)
        torch.save(standard_model.state_dict(), "{}/standard_cifar.pth.tar".format(save_dir))


if __name__ == "__main__":
    train_cifar(["rmaggnet", "ensemble", "ccat", "surrogate"], save_dir="trained_models")