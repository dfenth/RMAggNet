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
from arch.rm_models import RMEMNISTModel as RMModel
from arch.ensemble_models import EnsEMNISTModel as EnsModel
from arch.standard_models import EMNISTModel as StandardModel


def save_image_sample(sample_images, path):
    fig, axes = plt.subplots(nrows=1, ncols=sample_images.shape[0], constrained_layout=True)

    for i, img in enumerate(sample_images):
        img = np.squeeze(img)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    path = path.replace(" ", "").replace("(", "").replace(")", "")
    fig.savefig(path)
    plt.close()


def attack_emnist(models, attacks, type):
    """
    Evaluates the effect of adversarial attacks on the EMNIST dataset

    Parameters:
    - models (dict): A dictionary of models to train (choose from 'rmaggnet', 'ensemble' and 'ccat') as keys with the file names as values
    - attacks (list of string): A list of attacks to use (choose from 'pgdl2', 'pgdlinf', 'cwl2' and 'boundary')
    - type (list of string): A list of attack types (choose from 'openbox', 'closedbox')
    """
    #### Set up the logger
    logname = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")
    logging.basicConfig(level=logging.INFO, filename="{}_adv_emnist.log".format(logname))
    logger = logging.getLogger(__name__)
    ####

    batch_size = 128
    adv_sample_size = 1000

    # Load EMNIST (131,600 samples with 47 balanced classes)
    # www.westernsydney.edu.au/__data/assets/text_file/0019/1204408/EMNIST_Readme.txt
    train_mnist = torchvision.datasets.EMNIST('/tmp', split='balanced', train=True, download=True)
    test_mnist = torchvision.datasets.EMNIST('/tmp', split='balanced', train=False, download=True)

    train_dataset_full = transform_mnist_dataset(train_mnist)
    val_ratio = 0.1
    train_dataset = train_dataset_full[:int(len(train_dataset_full)-len(train_dataset_full)*val_ratio)]
    validation_dataset = train_dataset_full[int(len(train_dataset_full)-len(train_dataset_full)*val_ratio):]
    test_dataset = transform_mnist_dataset(test_mnist)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make sure learning rates etc. match how the model was trained!
    ################################# RMAggNet
    if 'rmaggnet' in models:
        rm_aggnet = RMAggNet([x for x in range(47)], RMModel, m=5, r=1, load_path="trained_models/{}".format(models['rmaggnet']))

        ################################# RMAggDiff
        # A differentiable approximation to RMAggNet
        hybrid = RMAggDiff(rm_aggnet)


    ################################# Ensemble
    if 'ensemble' in models:
        ensemble_model = ensemble.Ensemble(EnsModel, 32, load_path="trained_models/{}".format(models['ensemble']))


    ################################# CCAT
    if 'ccat' in models:
        # Setup a model, the optimizer, learning rate scheduler.
        ccat_base_model = EnsModel()
        ccat = CCAT(ccat_base_model)
        ccat.ccat_model.load_state_dict(torch.load("trained_models/{}.pth.tar".format(models['ccat'])))


    #################################################### Evaluation!
    thresholds = [x/10 for x in range(11)]
    logger.info("=== Clean data ===")

    if 'ccat' in models:
        logger.info("= CCAT =\nCorrect | Rejected | Incorrect")
        ccat.evaluate(test_loader, confidence_thresholds=thresholds, cuda=True, logger=logger)

    if 'ensemble' in models:
        logger.info("= Ensemble =\nCorrect | Rejected | Incorrect")
        ensemble.ensemble_eval(ensemble_model, test_loader, thresholds=thresholds, logger=logger)

    if 'rmaggnet' in models:
        logger.info("= RMAggNet =\nCorrect | Rejected | Incorrect")
        aggnet_eval(rm_aggnet, test_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=7, logger=logger)

        logger.info("= RMAggDiff =")
        res = hybrid.evaluate(test_loader)
        logger.info(res)


    #################################################### Random Noise OOD data
    logger.info("=== Random Noise (OOD) ===")
    noise_images = torch.rand(size=(10000, 1, 28, 28))
    noise_labels = np.random.randint(0, 47, size=(10000))
    random_dataset = torch.utils.data.DataLoader(list(zip(noise_images, noise_labels)), batch_size=batch_size, shuffle=False)

    if 'ccat' in models:
        logger.info("= CCAT =\nRejected | Incorrect")
        ccat.evaluate(random_dataset, confidence_thresholds=thresholds, cuda=True, logger=logger, ood=True)

    if 'ensemble' in models:
        logger.info("= Ensemble =\nRejected | Incorrect")
        ensemble.ensemble_eval(ensemble_model, random_dataset, thresholds=thresholds, logger=logger, ood=True)

    if 'rmaggnet' in models:
        logger.info("= RMAggNet =\nRejected | Incorrect")
        aggnet_eval(rm_aggnet, list(zip(noise_images, noise_labels)), batch_size=batch_size, thresholds=[0.5], max_correction=7, logger=logger, ood=True)

        logger.info("= RMAggDiff =")
        res = hybrid.evaluate(random_dataset)
        logger.info(res)


    ############################################ Adversarial attacks!

    test_images = torch.cat(list(map(lambda x: torch.unsqueeze(x[0], dim=0), test_dataset[:adv_sample_size])), dim=0)
    test_labels = torch.tensor(list(map(lambda x: x[1], test_dataset[:adv_sample_size])))
    test_images = test_images.to('cuda')
    test_labels = test_labels.to('cuda')

    # Load surrogate model for closed-box adversarial attack via transferability
    logger.info("=== Surrogate model ===")
    base_model = StandardModel()
    base_model.load_state_dict(torch.load("trained_models/standard_emnist.pth.tar"))
    base_model.to('cuda')
    base_model.evaluate(test_loader, logger=logger)

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
        logger.info(attack_name)
        
        if 'closedbox' in type:
            #### Transfer attacks
            logger.info("=== Transfer attacks ===")
            
            adv_surrogate_model = fb.PyTorchModel(base_model, bounds=(0,1))
            raw_adv, clipped_adv_eps, is_adv = attack(adv_surrogate_model, test_images, test_labels, epsilons=epsilons)
            
            # Save adversarial image samples for each epsilon
            for idx, ep in enumerate(epsilons):
                save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "base_EMNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))

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
                    ccat.evaluate(adv_loader, confidence_thresholds=thresholds, cuda=True, logger=logger)

            if 'ensemble' in models:
                logger.info("== Ensemble ==\nCorrect | Rejected | Incorrect")
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ensemble.ensemble_eval(ensemble_model, adv_loader, thresholds=thresholds, logger=logger)

            if 'rmaggnet' in models:
                logger.info("== RMAggNet ==\nCorrect | Rejected | Incorrect")
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_dataset = list(zip(adv_data, test_labels))
                    logger.info("- eps: {} -".format(eps))
                    aggnet_eval(rm_aggnet, adv_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=7, logger=logger)
        
        if 'openbox' in type:
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
                        save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "ccat_EMNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))
                
                # Evaluate on the adversarial images
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ccat.evaluate(adv_loader, confidence_thresholds=thresholds, cuda=True, logger=logger)
                    del adv_loader

            if 'ensemble' in models:
                logger.info("== Ensemble ==\nCorrect | Rejected | Incorrect")
                ensemble_model.to("cuda")
                adv_ensemble_model = fb.PyTorchModel(ensemble_model, bounds=(0,1))
                raw_adv, clipped_adv_eps, is_adv = attack(adv_ensemble_model, test_images, test_labels, epsilons=epsilons)
                
                # Save adversarial image samples for each epsilon
                for idx, ep in enumerate(epsilons):
                    save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "ens_EMNIST_{}_{}.png".format(attack_name, str(ep).replace(".","_")))
                
                # Evaluate on the adversarial images
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("- eps: {} -".format(eps))
                    ensemble.ensemble_eval(ensemble_model, adv_loader, thresholds=thresholds, logger=logger)
                    del adv_loader
                
                ensemble_model.to('cpu')

            if 'rmaggnet' in models:
                logger.info("== RMAggDiff ==\nCorrect | Rejected | Incorrect")
                adv_hybrid_model = fb.PyTorchModel(hybrid, bounds=(0,1))
                raw_adv, clipped_adv_eps, is_adv = attack(adv_hybrid_model, test_images, test_labels, epsilons=epsilons)

                # Save adversarial image samples for each epsilon
                for idx, ep in enumerate(epsilons):
                    save_image_sample(clipped_adv_eps[idx][:10].to('cpu'), "hybrid_EMNIST_{}_{}.png".format(attack_name.replace(" ", ""), str(ep).replace(".","_")))
                
                for eps, adv_data in zip(epsilons, clipped_adv_eps):
                    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, test_labels)), batch_size=batch_size, shuffle=False)
                    logger.info("== RMAggDiff ==")
                    logger.info("-- eps: {} --".format(eps))
                    logger.info(hybrid.evaluate(adv_loader))
                    
                    logger.info("== RMAggNet ==\nCorrect | Rejected | Incorrect")
                    logger.info("-- eps: {} --".format(eps))
                    tmp_test_labels = torch.tensor([l for _,l in test_dataset[:adv_sample_size]])
                    adv_dataset = list(zip(adv_data, tmp_test_labels))
                    aggnet_eval(rm_aggnet, adv_dataset, batch_size=batch_size, thresholds=[0.5], max_correction=7, logger=logger)
                    del adv_loader, adv_dataset, tmp_test_labels

if __name__ == "__main__":
    attack_emnist(
        models={
            "rmaggnet": "rmaggnet_emnist",
            "ensemble": "ens_emnist",
            "ccat": "ccat_emnist"
        },
        attacks=["pgdl2", "pgdlinf"],
        type=["closedbox", "openbox"]
    )