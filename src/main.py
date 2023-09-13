import argparse

import mnist_train, emnist_train, cifar_train
import adversarial_mnist_tests, adversarial_emnist_tests, adversarial_cifar_tests

parser = argparse.ArgumentParser(description="Allows modular execution of the Reed-Muller Aggregation Network code")

parser.add_argument(
    '--mode', 
    type=str, 
    help='Defines the execution mode (train or attack) which trains the specified models or performs an adversarial attack',
    choices=["train", "attack"],
    required=True
)

parser.add_argument(
    '--dataset',
    type=str,
    help='List all of the datasets to be included in the execution',
    choices=["mnist", "emnist", "cifar", "all"],
    nargs="+", # We expect one or more arguments!
    default='mnist',
    required=False
)

parser.add_argument(
    '--models',
    type=str,
    help='List all of the models to be included in the execution',
    choices=["rmaggnet", "ensemble", "ccat", "all"],
    nargs="+",
    default='all',
    required=False
)

parser.add_argument(
    '--boxtype',
    type=str,
    help='Select the box type (`openbox`, `closedbox` or `both`)',
    choices=['openbox', 'closedbox', 'both'],
    nargs="+",
    default='both',
    required=False   
)

parser.add_argument(
    '--attacks',
    type=str,
    help='List all attacks to be used if `--mode attack`',
    choices=['pgdl2', 'pgdlinf', 'cwl2', 'boundary', 'all'],
    nargs="+",
    default='all',
    required=False
)

parser.add_argument(
    '--dir',
    type=str,
    help='The directory to save the models to if `--mode train`, or load the models from if `--mode attack`',
    required=True
)

args = parser.parse_args()

datasets = []
if "all" in args.dataset:
    datasets = ["mnist", "emnist", "cifar"]
else:
    datasets.append(args.dataset)

if "all" in args.models:
    models = ["rmaggnet", "ensemble", "ccat", "surrogate"]
else:
    models = args.models
    models.append("surrogate")


if args.mode == "train":
    print("Training!")

    for d in datasets:
        if d == 'mnist':
            print("Training MNIST: {}, {}".format(models, args.dir))
            mnist_train.train_mnist(models, args.dir)
        elif d == 'emnist':
            print("Training EMNIST: {}, {}".format(models, args.dir))
            emnist_train.train_emnist(models, args.dir)
        elif d == 'cifar':
            print("Training CIFAR: {}, {}".format(models, args.dir))
            cifar_train.train_cifar(models, args.dir)

elif args.mode == "attack":
    print("Attacking!")

    if 'all' in args.attacks:
        attacks = ['pgdlinf', 'pgdl2', 'cwl2', 'boundary']
    else:
        attacks = args.attacks
    
    if 'both' in args.boxtype:
        boxtype = ['openbox', 'closedbox']
    else:
        boxtype = args.boxtype

    for d in datasets:
        if d == 'mnist':
            print("Attacking MNIST: {}, {}, {}, {}".format(models, args.dir, attacks, boxtype))
            adversarial_mnist_tests.attack_mnist(models, args.dir, attacks, type)
        elif d == 'emnist':
            print("Attacking EMNIST: {}, {}, {}, {}".format(models, args.dir, attacks, boxtype))
            adversarial_emnist_tests.attack_emnist(models, args.dir, attacks, type)
        elif d == 'cifar':
            print("Attacking CIFAR: {}, {}, {}, {}".format(models, args.dir, attacks, boxtype))
            adversarial_cifar_tests.attack_cifar(models, args.dir, attacks, type)

else:
    print("Failed to execute due to bad --mode")