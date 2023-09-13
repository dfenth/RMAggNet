import torch
import torchvision
import numpy as np
import datetime
import logging

from matplotlib import pyplot as plt

import foolbox as fb

from rm_aggnet import ReedMullerAggregationNetwork as RMAggNet 
from rm_aggnet import aggnet_eval
import ensemble
from ccat_model import CCAT
from rma_diff import RMAggDiff

from utils.dataset_transform import transform_mnist_dataset

# Import the models
from arch.rm_models import RMMNISTModel as RMModel
from arch.ensemble_models import EnsMNISTModel as EnsModel
from arch.standard_models import MNISTModel as StandardModel 


def save_image_sample(sample_images, path):
    fig, axes = plt.subplots(nrows=1, ncols=sample_images.shape[0], constrained_layout=True)

    for i, img in enumerate(sample_images):
        img = np.squeeze(img)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    path = path.replace(" ", "").replace("(", "").replace(")", "")
    fig.savefig(path)
    plt.close()


def attack_mnist(models, load_dir, attacks, attack_type, cuda):
    """
    Evaluates the effect of adversarial attacks on the MNIST dataset

    Parameters:
    - models (list of string): A list of models to train (choose from 'rmaggnet', 'ensemble', 'ccat' and 'surrogate')
    - load_dir (string): The directory to load the models from
    - attacks (list of string): A list of attacks to use (choose from 'pgdl2', 'pgdlinf', 'cwl2' and 'boundary')
    - attack_type (list of string): A list of attack types (choose from 'openbox', 'closedbox')
    - cuda (bool): Whether to run with CUDA
    """
    #### Set up the logger
    logname = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")
    logging.basicConfig(level=logging.INFO, filename="{}_adv_mnist.log".format(logname))
    logger = logging.getLogger(__name__)
    ####

    device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

    batch_size = 256
    adv_sample_size = 1000

    # Load the MNIST dataset
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

    # Make sure learning rates etc. match how the model was trained!
    ################################# RMAggNet
    if 'rmaggnet' in models:
        rm_aggnet = RMAggNet([x for x in range(10)], RMModel, m=4, r=1, load_path="{}/rmaggnet_mnist".format(load_dir), cuda=cuda)
        

        ################################# RMAggDiff
        # A differentiable approximation to RMAggNet
        hybrid = RMAggDiff(rm_aggnet)


    ################################# Ensemble
    if 'ensemble' in models:
        ensemble_model = ensemble.Ensemble(EnsModel, 16, load_path="{}/ens_mnist".format(load_dir))


    ################################# CCAT
    if 'ccat' in models:
        # Setup a model, the optimizer, learning rate scheduler.
        ccat_base_model = EnsModel()
        ccat = CCAT(ccat_base_model)
        ccat.ccat_model.load_state_dict(torch.load("{}/ccat_mnist.pth.tar".format(load_dir)))
        


    #################################################### Evaluation!
    thresholds = [x/10 for x in range(11)]
    logger.info("=== Clean data ===")

    if 'ccat' in models:
        logger.info("= CCAT =\nCorrect | Rejected | Incorrect")
        ccat.evaluate(test_loader, confidence_thresholds=thresholds, cuda=cuda, logger=logger)

    if 'ensemble' in models:
        logger.info("= Ensemble =\nCorrect | Rejected | Incorrect")
        ensemble.ensemble_eval(ensemble_model, test_loader, thresholds=thresholds, logger=logger, cuda=cuda)

    if 'rmaggnet' in models:
        logger.info("= RMAggNet =\nCorrect | Rejected | Incorrect")
        aggnet_eval(rm_aggnet, test_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=logger)

        logger.info("= RMAggDiff =")
        res = hybrid.evaluate(test_loader, cuda=cuda)
        logger.info(res)

    #################################################### Random Noise OOD data
    logger.info("=== Random Noise (OOD) ===")
    noise_images = torch.rand(size=(10000, 1, 28, 28))
    noise_labels = np.random.randint(0, 10, size=(10000))
    random_dataset = torch.utils.data.DataLoader(list(zip(noise_images, noise_labels)), batch_size=batch_size, shuffle=False)

    if 'ccat' in models:
        logger.info("= CCAT =\nRejected | Incorrect")
        ccat.evaluate(random_dataset, confidence_thresholds=thresholds, cuda=cuda, logger=logger, ood=True)

    if 'ensemble' in models:
        logger.info("= Ensemble =\nRejected | Incorrect")
        ensemble.ensemble_eval(ensemble_model, random_dataset, thresholds=thresholds, logger=logger, ood=True, cuda=cuda)

    if 'rmaggnet' in models:
        logger.info("= RMAggNet =\nRejected | Incorrect")
        aggnet_eval(rm_aggnet, list(zip(noise_images, noise_labels)), batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=logger, ood=True)

        logger.info("= RMAggDiff =")
        res = hybrid.evaluate(random_dataset, cuda=cuda)
        logger.info(res)


    #################################################### Fashion MNIST OOD data
    logger.info("=== Fashion MNIST (OOD) ===")
    fmnist_test = torchvision.datasets.FashionMNIST('/tmp', train=False, download=True)
    fmnist_dataset = transform_mnist_dataset(fmnist_test)
    fmnist_loader = torch.utils.data.DataLoader(fmnist_dataset, batch_size=batch_size, shuffle=True)

    if 'ccat' in models:
        logger.info("= CCAT =\nRejected | Incorrect")
        ccat.evaluate(fmnist_loader, confidence_thresholds=thresholds, cuda=cuda, logger=logger, ood=True)
    
    if 'ensemble' in models:
        logger.info("= Ensemble =\nRejected | Incorrect")
        ensemble.ensemble_eval(ensemble_model, fmnist_loader, thresholds=thresholds, logger=logger, ood=True, cuda=cuda)

    if 'rmaggnet' in models:
        logger.info("= RMAggNet =\nRejected | Incorrect")
        aggnet_eval(rm_aggnet, fmnist_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=logger, ood=True)

        logger.info("= RMAggDiff =")
        res = hybrid.evaluate(fmnist_loader)
        logger.info(res)


    ############################################ Adversarial attacks!

    test_images = torch.cat(list(map(lambda x: torch.unsqueeze(x[0], dim=0), test_dataset[:adv_sample_size])), dim=0)
    test_labels = torch.tensor(list(map(lambda x: x[1], test_dataset[:adv_sample_size])))
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    # Load surrogate model for closed-box adversarial attack via transferability
    logger.info("=== Surrogate model ===")
    base_model = StandardModel()
    base_model.load_state_dict(torch.load("{}/standard_mnist.pth.tar".format(load_dir)))
    base_model.to(device)
    base_model.evaluate(test_loader, logger=logger, cuda=cuda)

    adversarial_attacks = {}

    # Populate the attack list based on function input
    if 'pgdlinf' in attacks:
        adversarial_attacks["PGD (Linf)"] = [0, 0.05, 0.1, 0.3, 0.6, 1.0]
    if 'pgdl2' in attacks:
        adversarial_attacks["PGD (L2)"] = [0, 0.3, 0.5, 0.75, 1, 2.5, 3]
    if 'cwl2' in attacks:
        adversarial_attacks["C&W L2"] = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    if 'boundary' in attacks:
        adversarial_attacks["Boundary (L2)"] = [1.5, 2, 2.5, 3]    

    for attack_name in adversarial_attacks:

        if attack_name == "PGD (Linf)":
            attack = fb.attacks.LinfProjectedGradientDescentAttack()
        elif attack_name == "PGD (L2)":
            attack = fb.attacks.L2ProjectedGradientDescentAttack()
        elif attack_name == "C&W L2":
            attack = fb.attacks.L2CarliniWagnerAttack(confidence=0)
        elif attack_name == "Boundary (L2)":
            init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(directions=5000, steps=2000)
            attack = fb.attacks.BoundaryAttack(init_attack=init_attack)
        else:
            logger.info("Cannot recognise attack: {}".format(attack_name))
            exit()
        
        epsilons = adversarial_attacks[attack_name]
        logger.info("***** {} Attack *****".format(attack_name))
        
        if 'closedbox' in attack_type:
            #### Transfer attacks
            logger.info("=== Transfer attacks ===")
            
            adv_surrogate_model = fb.PyTorchModel(base_model, bounds=(0,1))
            raw_adv, clipped_adv_eps, is_adv = attack(adv_surrogate_model, test_images, test_labels, epsilons=epsilons)
            
            # Save adversarial image samples for each epsilon
            for idx, ep in enumerate(epsilons):
                save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "base_MNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))

            # Evaluate the success of the transfer attack
            logger.info("== Base model (open-box) ==")
            for eps, adv_data in zip(epsilons, clipped_adv_eps):
                adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                logger.info("- eps: {} -".format(eps))
                base_model.evaluate(adv_loader, logger=logger)
            
            if 'ccat' in models:
                logger.info("== CCAT ==\nCorrect | Rejected | Incorrect")
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ccat.evaluate(adv_loader, confidence_thresholds=thresholds, cuda=cuda, logger=logger)

            if 'ensemble' in models:
                logger.info("== Ensemble ==\nCorrect | Rejected | Incorrect")
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ensemble.ensemble_eval(ensemble_model, adv_loader, thresholds=thresholds, logger=logger, cuda=cuda)

            if 'rmaggnet' in models:
                logger.info("== RMAggNet ==\nCorrect | Rejected | Incorrect")
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_dataset = list(zip(adv_data, test_labels))
                    logger.info("- eps: {} -".format(eps))
                    aggnet_eval(rm_aggnet, adv_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=logger)
        
        if 'openbox' in attack_type:
            #### Open-box attacks
            logger.info("=== Open-box attacks ===")
            
            if 'ccat' in models:
                logger.info("== CCAT ==\nCorrect | Rejected | Incorrect")
                adv_ccat_model = fb.PyTorchModel(ccat.ccat_model, bounds=(0,1))
                if "Boundary" in attack_name:
                    # Run the boundary attacks for individual eps values so we can handle failures
                    clipped_adv_eps = []
                    for eps in epsilons:
                        try:
                            _, clipped_advs, _ = attack(adv_ccat_model, test_images, test_labels, epsilons=eps)
                            clipped_adv_eps.append(clipped_advs)
                        except Exception as e:
                            logger.info("Adversarial attack Boundary (L2) failed for epsilon={}".format(eps))
                            logger.info("{}".format(e))
                else:
                    raw_adv, clipped_adv_eps, is_adv = attack(adv_ccat_model, test_images, test_labels, epsilons=epsilons)
                
                # Save adversarial image samples for each epsilon
                if len(clipped_adv_eps) > 0:
                    for idx, ep in enumerate(epsilons):
                        save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "ccat_MNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))
                
                # Evaluate on the adversarial images
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ccat.evaluate(adv_loader, confidence_thresholds=thresholds, cuda=cuda, logger=logger)
                    del adv_loader

            if 'ensemble' in models:
                logger.info("== Ensemble ==\nCorrect | Rejected | Incorrect")
                ensemble_model.to_device(device)

                adv_ensemble_model = fb.PyTorchModel(ensemble_model, bounds=(0,1))
                raw_adv, clipped_adv_eps, is_adv = attack(adv_ensemble_model, test_images, test_labels, epsilons=epsilons)
                
                # Save adversarial image samples for each epsilon
                for idx, ep in enumerate(epsilons):
                    save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "ens_MNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))
                
                # Evaluate on the adversarial images
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ensemble.ensemble_eval(ensemble_model, adv_loader, thresholds=thresholds, logger=logger)
                    del adv_loader
                
                ensemble_model.to_device('cpu')

            if 'rmaggnet' in models:
                hybrid.to_device(device=device)
                print("DR DEVICE:", next(hybrid.diff_replacement.parameters()).device)
                adv_hybrid_model = fb.PyTorchModel(hybrid, bounds=(0,1))
                raw_adv, clipped_adv_eps, is_adv = attack(adv_hybrid_model, test_images, test_labels, epsilons=epsilons)

                # Save adversarial image samples for each epsilon
                for idx, ep in enumerate(epsilons):
                    save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "hybrid_MNIST_{}_{}.png".format(attack_name.replace(" ", ""), str(ep).replace(".","_")))
                
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("== RMAggDiff ==\nCorrect | Rejected | Incorrect")
                    logger.info("-- eps: {} --".format(eps))
                    logger.info(hybrid.evaluate(adv_loader))
                    
                    logger.info("== RMAggNet ==\nCorrect | Rejected | Incorrect")
                    logger.info("-- eps: {} --".format(eps))
                    tmp_test_labels = torch.tensor([l for _,l in test_dataset[:adv_sample_size]])
                    adv_dataset = list(zip(adv_data, tmp_test_labels))
                    aggnet_eval(rm_aggnet, adv_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=3, logger=logger)
                    del adv_loader, adv_dataset, tmp_test_labels
                
                hybrid.to_device(device='cpu')

if __name__=="__main__":
    attack_mnist(
        models=[
        "rmaggnet",
        "ensemble",
        "ccat"
    ],
    load_dir="trained_models",
    attacks=["pgdl2", "pgdlinf"],
    attack_type=["closedbox", "openbox"],
    cuda=True
    )